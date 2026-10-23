"""Literature pair validation (analysis 08; see revision_analyses/README.md).

Input: literature/Cr_F_LiRich_primary_verified_samples_v1_20261023.csv.
Uses only the saved CatBoost full model and the 100 saved bootstrap models; nothing is
trained.  For every paper the undoped sample is the baseline; the measured and
predicted changes (doped - undoped) are compared in sign, and the 100 bootstrap
models give the P05-P95 interval of the predicted change.

Test conditions (from the CSV):
  Lee 2015     first cycle 10 mA/g, ambient T -> 298.15 K.  1C is not defined in the
               paper, so 1C = 200 / 250 / 300 mAh/g (0.05 / 0.04 / 0.0333 C) are all run.
  Wang 2024    0.1C, T not reported -> 298.15 K; later: cycle 50 at 0.1C.
  Kucuk 2026   C/10 (paper defines 1C = 250 mAh/g), T not reported -> 298.15 K;
               later: cycle 3 (Table 2 says C/3, Methods say C/10: both run) and cycle 100 at C/3.
  Im 2019      C/10 (1C = 230 mA/g), reported separately (spinel-layered).
The dataset itself keeps only records whose current unit is C (Step11
split_info_current_cycle.py keeps current_unit == 'c'), so C-rates are used as reported.
Outputs: outputs/08_*.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
from pymatgen.core import Composition

import followup_checks as f6
from composition_counterparts import load_models
from candidate_checks import OUT, log, ssb, write_json

SAMPLES = f6.LITERATURE.parent / "Cr_F_LiRich_primary_verified_samples_v1_20261023.csv"
T_DEFAULT = 298.15

# (paper key, label, measured column, cycle, C-rate) - one entry per comparison scenario
SCENARIOS = [
    ("Lee", "首圈放電，1C = 200 mAh/g", "first", 1, 10 / 200),
    ("Lee", "首圈放電，1C = 250 mAh/g", "first", 1, 10 / 250),
    ("Lee", "首圈放電，1C = 300 mAh/g", "first", 1, 10 / 300),
    ("Wang", "首圈放電，0.1C", "first", 1, 0.1),
    ("Wang", "第 50 圈放電，0.1C", "cycle50", 50, 0.1),
    ("Kucuk", "首圈放電，C/10", "first", 1, 0.1),
    ("Kucuk", "第 3 圈放電，C/3（依 Table 2）", "cycle3", 3, 1 / 3),
    ("Kucuk", "第 3 圈放電，C/10（依 Methods）", "cycle3", 3, 0.1),
    ("Kucuk", "第 100 圈放電，C/3", "cycle100", 100, 1 / 3),
    ("Im", "首圈放電，C/10", "first", 1, 0.1),
]


def number(text: str) -> float:
    try:
        return float(str(text).strip())
    except ValueError:
        return math.nan


def load_samples() -> pd.DataFrame:
    lit = pd.read_csv(SAMPLES, encoding="utf-8-sig", dtype=str)
    rows = []
    for _, r in lit.iterrows():
        formula = f6.literature_formula(r["論文中名義化學式_保留原文參數寫法"], r["摻雜量_原文x"])
        later = str(r["活化後或第N圈放電容量_mAh_g"])
        cycle3 = cycle100 = cycle50 = math.nan
        found = dict(re.findall(r"第(\d+)圈\s+(\d+(?:\.\d+)?)", later))  # Kucuk: "第3圈 149；第100圈 100"
        if found:
            cycle3, cycle100 = float(found.get("3", "nan")), float(found.get("100", "nan"))
        elif str(r["活化後或第N圈的循環次數"]).strip() == "50":  # Wang
            cycle50 = number(later)
        rows.append({
            "paper": r["作者_簡稱"].split()[0], "DOI": r["DOI"], "sample": r["樣品名"],
            "undoped": r["是否未摻雜對照"].strip() == "是", "nominal": r["論文中名義化學式_保留原文參數寫法"],
            "Formula": formula, "canonical_key": ssb.canonical_key(Composition(formula)),
            "first": number(r["首圈放電容量_mAh_g"]), "cycle3": cycle3, "cycle50": cycle50, "cycle100": cycle100,
        })
    return pd.DataFrame(rows)


def features(evaluator: ssb.CandidateEvaluator, formulas: list[str], rate: float, cycle: float) -> pd.DataFrame:
    rows = []
    for formula in formulas:
        comp = Composition(formula)
        rows.append(evaluator._composition_row({"comp": comp.get_el_amt_dict(), "Formula": formula,
                                                 "canonical_key": ssb.canonical_key(comp)}))
    X = pd.DataFrame(rows).reindex(columns=evaluator.comp_columns)
    X["temperature_K"] = T_DEFAULT
    X["log1p_current_rate_C"] = math.log1p(rate)
    X["log1p_cycle_number"] = math.log1p(cycle)
    X["log1p_current_rate_x_cycle"] = math.log1p(rate * cycle)
    return X.reindex(columns=evaluator.feature_columns)


def main() -> None:
    evaluator = ssb.CandidateEvaluator()
    models, feature_columns = load_models()
    if feature_columns != list(evaluator.feature_columns):
        raise RuntimeError("Bootstrap model feature order differs from the evaluator")
    train_keys = set(evaluator.unique_train.index)
    samples = load_samples()
    samples["in_training"] = samples["canonical_key"].isin(train_keys)
    samples.to_csv(OUT / "08_samples.csv", index=False, encoding="utf-8-sig")
    log(f"samples in training: {samples['in_training'].sum()} of {len(samples)}")

    pred_rows, pair_rows = [], []
    for paper, label, column, cycle, rate in SCENARIOS:
        part = samples[samples["paper"].eq(paper)].reset_index(drop=True)
        X = features(evaluator, part["Formula"].tolist(), rate, cycle)
        full = np.asarray(evaluator.model.predict(X, thread_count=1), dtype=float)
        boot = np.column_stack([np.asarray(m.predict(X, thread_count=1), dtype=float) for m in models])
        for i, r in part.iterrows():
            pred_rows.append({"paper": paper, "scenario": label, "sample": r["sample"], "Formula": r["Formula"],
                              "C_rate": rate, "cycle": cycle, "measured": r[column], "pred_full": full[i],
                              "pred_full_clipped": float(np.clip(full[i], *evaluator.prediction_clip)),
                              "boot_P05": float(np.percentile(boot[i], 5)), "boot_P95": float(np.percentile(boot[i], 95)),
                              "in_training": bool(r["in_training"])})
        base_idx = part.index[part["undoped"]]
        if len(base_idx) != 1:
            raise RuntimeError(f"{paper}: expected one undoped baseline")
        b = int(base_idx[0])
        for i, r in part.iterrows():
            if i == b:
                continue
            measured_delta = r[column] - part.loc[b, column]
            delta_boot = boot[i] - boot[b]
            p05, p95 = float(np.percentile(delta_boot, 5)), float(np.percentile(delta_boot, 95))
            pred_delta = full[i] - full[b]
            pair_rows.append({
                "paper": paper, "scenario": label, "doped": r["sample"], "doped_formula": r["Formula"],
                "baseline_formula": part.loc[b, "Formula"],
                "doped_in_training": bool(r["in_training"]), "baseline_in_training": bool(part.loc[b, "in_training"]),
                "measured_baseline": part.loc[b, column], "measured_doped": r[column],
                "measured_delta": measured_delta, "pred_delta_full": pred_delta,
                "boot_delta_P05": p05, "boot_delta_P95": p95, "boot_interval_covers_0": bool(p05 <= 0 <= p95),
                "boot_frac_same_sign_as_measured": (float(np.mean(np.sign(delta_boot) == np.sign(measured_delta)))
                                                    if np.isfinite(measured_delta) and measured_delta != 0 else math.nan),
                "measured_available": bool(np.isfinite(measured_delta)),
                "direction_agrees": (bool(np.sign(measured_delta) == np.sign(pred_delta))
                                     if np.isfinite(measured_delta) and measured_delta != 0 else None),
            })
    preds = pd.DataFrame(pred_rows)
    pairs = pd.DataFrame(pair_rows)
    pairs["excluded_overlap"] = pairs["doped_in_training"] | pairs["baseline_in_training"]
    preds.to_csv(OUT / "08_predictions.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(OUT / "08_pairs.csv", index=False, encoding="utf-8-sig")
    write_json("08_summary.json", {"n_samples": len(samples), "n_in_training": int(samples["in_training"].sum()),
                                   "in_training": samples[samples["in_training"]][["paper", "sample", "Formula"]]
                                   .to_dict(orient="records"),
                                   "prediction_clip": evaluator.prediction_clip})


if __name__ == "__main__":
    main()
