"""Literature pair validation on the v2 sample table (analysis 08b; see revision_analyses/README.md).

Same method as literature_pairs.py, whose functions are reused unchanged:
  - only the saved CatBoost full model and the 100 saved bootstrap models; nothing is trained;
  - T = 298.15 K, C-rate as reported by the paper, cycle as reported;
  - baseline = the paper's undoped sample; measured and predicted changes (doped - undoped) compared in sign;
    bootstrap P05-P95 of the paired change;
  - a pair is excluded from the main statistics when either composition is one of the 171 training
    compositions (canonical_key);
  - the source-paper overlap check of analysis 08 section 6 (09 traced records, 2025 verification PDFs,
    BDE clean-up tables) is repeated for every DOI in the v2 table.

Input: literature/Cr_F_LiRich_primary_verified_samples_v2_20261024.csv (v1 17 rows + Kim 2013 7 rows +
Pang 2017 7 rows; built by literature_v2_20261024.py).

Model formulas for the new rows (the CSV keeps the papers' own wording):
  Kim 2013   precursor mole ratio Li : Mn : M = 2 : 0.99 : 0.01 -> Li2Mn0.99M0.01O3; Cr 2 / 5 mol% (written
             Li2MnxCr1-xO3) -> Li2Mn0.98Cr0.02O3 / Li2Mn0.95Cr0.05O3.  First cycle 10 mA/g, 1C not defined ->
             1C = 200 / 250 / 300 mAh/g, as for Lee 2015 in 08.
  Pang 2017  LNCM Li1.2Ni0.23Co0.12Mn0.5O2.  "5% F" -> O1.9F0.1 (the refined NPD formula of the layered phase
             is O1.90(6)F0.10(6)).  "2% Cr" has no stated basis -> run twice: Cr0.02 added to LNCM (main), and
             Cr replacing 2% of every transition metal (Li1.2Ni0.2254Co0.1176Mn0.49Cr0.017O2).
             The four preliminary samples of SI Fig. S1 have no undoped value and are not paired.
Outputs: outputs/08b_*.
"""
from __future__ import annotations

import json
import math
import re

import numpy as np
import pandas as pd
from pymatgen.core import Composition

import followup_checks as f6
import literature_pairs as l8
from composition_counterparts import load_models
from training_source_audit import BI, DOI_TABLES
from candidate_checks import OUT, log, ssb, write_json

SAMPLES = f6.LITERATURE.parent / "Cr_F_LiRich_primary_verified_samples_v2_20261024.csv"

MODEL_FORMULA = {
    ("Kim", "Li2MnO3（pristine）"): "Li2MnO3",
    ("Kim", "Al-Li2MnO3"): "Li2Mn0.99Al0.01O3",
    ("Kim", "Cr-Li2MnO3（1 mol%）"): "Li2Mn0.99Cr0.01O3",
    ("Kim", "Fe-Li2MnO3"): "Li2Mn0.99Fe0.01O3",
    ("Kim", "Zn-Li2MnO3"): "Li2Mn0.99Zn0.01O3",
    ("Kim", "Cr-Li2MnO3（2 mol%）"): "Li2Mn0.98Cr0.02O3",
    ("Kim", "Cr-Li2MnO3（5 mol%）"): "Li2Mn0.95Cr0.05O3",
    ("Pang", "LNCM"): "Li1.2Ni0.23Co0.12Mn0.5O2",
    ("Pang", "Cr-LNCM"): "Li1.2Ni0.23Co0.12Mn0.5Cr0.02O2",
    ("Pang", "F-LNCM"): "Li1.2Ni0.23Co0.12Mn0.5O1.9F0.1",
}
PANG_CR_SUBSTITUTED = "Li1.2Ni0.2254Co0.1176Mn0.49Cr0.017O2"
PANG_PAIRED = ["LNCM", "Cr-LNCM", "F-LNCM"]
# Kim p164: "the charge and discharge capacities of Li2MnO3 decreased to 223.2 ... 191.1" - charge or discharge
# is not stated, so these are used only in a separately labelled supplementary scenario.
KIM_UNSPECIFIED = {"Cr-Li2MnO3（2 mol%）": 223.2, "Cr-Li2MnO3（5 mol%）": 191.1}
FULL_CELL_CYCLES = [10, 50, 100, 150, 200]

# (paper, label, measured column, cycle, C-rate, Pang Cr formula) - 08's scenarios first, unchanged
SCENARIOS = [(p, label, col, cyc, rate, None) for p, label, col, cyc, rate in l8.SCENARIOS] + [
    ("Kim", "首圈放電，1C = 200 mAh/g", "first", 1, 10 / 200, None),
    ("Kim", "首圈放電，1C = 250 mAh/g", "first", 1, 10 / 250, None),
    ("Kim", "首圈放電，1C = 300 mAh/g", "first", 1, 10 / 300, None),
    ("Kim", "補充：Cr 2／5 mol% 的 223.2／191.1（原文未說明充或放電），1C = 250 mAh/g", "kim_unspecified", 1, 10 / 250, None),
    ("Pang", "首圈放電，0.1C", "first", 1, 0.1, "added"),
    ("Pang", "首圈放電，0.1C（Cr 取代 2% 過渡金屬）", "first", 1, 0.1, "substituted"),
    ("Pang", "第 2 圈放電，0.1C", "cycle2", 2, 0.1, "added"),
    *[("Pang", f"全電池第 {n} 圈放電，{rate_label}", f"full{n}", n, rate, "added")
      for n in FULL_CELL_CYCLES for rate_label, rate in [("0.5C（放電速率）", 0.5), ("0.2C（充電速率）", 0.2)]],
]


def load_samples() -> pd.DataFrame:
    lit = pd.read_csv(SAMPLES, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    rows = []
    for _, r in lit.iterrows():
        paper, sample = r["作者_簡稱"].split()[0], r["樣品名"]
        later, other = str(r["活化後或第N圈放電容量_mAh_g"]), str(r["其他已報告容量與條件"])
        cyc = {f"cycle{n}": float(v) for n, v in re.findall(r"第(\d+)圈\s+(\d+(?:\.\d+)?)", later)}
        cyc.update({f"full{n}": float(v) for n, v in re.findall(r"全電池第(\d+)圈\s+(\d+(?:\.\d+)?)", other)})
        if paper == "Wang" and str(r["活化後或第N圈的循環次數"]).strip() == "50":  # as in 08
            cyc["cycle50"] = l8.number(later)
        if paper in ("Kim", "Pang"):
            formula = MODEL_FORMULA.get((paper, sample))
        else:
            formula = f6.literature_formula(r["論文中名義化學式_保留原文參數寫法"], r["摻雜量_原文x"])
        first = l8.number(r["首圈放電容量_mAh_g"])
        rows.append({
            "paper": paper, "DOI": r["DOI"], "sample": sample, "undoped": r["是否未摻雜對照"].strip() == "是",
            "dopant": r["摻雜元素"], "nominal": r["論文中名義化學式_保留原文參數寫法"], "Formula": formula,
            "canonical_key": ssb.canonical_key(Composition(formula)) if formula else None,
            "first": first, "kim_unspecified": KIM_UNSPECIFIED.get(sample, first if paper == "Kim" else math.nan),
            **{c: cyc.get(c, math.nan) for c in ["cycle2", "cycle3", "cycle50", "cycle100"]},
            **{f"full{n}": cyc.get(f"full{n}", math.nan) for n in FULL_CELL_CYCLES},
        })
    return pd.DataFrame(rows)


def doi_overlap(dois: list[str]) -> pd.DataFrame:
    """Source-paper overlap, same rule as analysis 08 section 6 (outputs/11_training_overlap.csv)."""
    rec = pd.read_csv(OUT / "09_records_907.csv", encoding="utf-8-sig")
    rec["tier"] = rec.trace_status.str[:2].str.strip()
    pdf_set = {str(d).lower() for d in pd.read_csv(OUT / "09_pdf_dois.csv", encoding="utf-8-sig").DOI.dropna()}
    pool: set[str] = set()
    for rel in DOI_TABLES.values():
        table = pd.read_csv(BI / rel, low_memory=False)
        pool |= {x.strip().lower() for s in table.DOI.dropna().astype(str) for x in s.split(",") if x.strip()}
    rows = []
    for doi in dois:
        low = doi.lower()
        hit = rec[rec.DOI.astype(str).str.lower().str.split("; ").apply(lambda xs, low=low: low in xs)]
        rows.append({"DOI": doi, "n_records_any_tier": len(hit),
                     "tiers": json.dumps(hit.tier.value_counts().to_dict(), ensure_ascii=False),
                     "in_verification_pdfs": low in pdf_set, "in_BDE_pipeline_tables": low in pool})
    return pd.DataFrame(rows)


def main() -> None:
    evaluator = ssb.CandidateEvaluator()
    models, feature_columns = load_models()
    if feature_columns != list(evaluator.feature_columns):
        raise RuntimeError("Bootstrap model feature order differs from the evaluator")
    train_keys = set(evaluator.unique_train.index)
    samples = load_samples()
    samples["in_training"] = samples["canonical_key"].isin(train_keys)
    samples.to_csv(OUT / "08b_samples.csv", index=False, encoding="utf-8-sig")
    sub_key = ssb.canonical_key(Composition(PANG_CR_SUBSTITUTED))
    log(f"samples in training: {samples['in_training'].sum()} of {len(samples)}; "
        f"Pang Cr substituted in training: {sub_key in train_keys}")

    pred_rows, pair_rows = [], []
    for paper, label, column, cycle, rate, cr_form in SCENARIOS:
        part = samples[samples["paper"].eq(paper) & samples["Formula"].notna()].reset_index(drop=True)
        if paper == "Pang":
            part = part[part["sample"].isin(PANG_PAIRED)].reset_index(drop=True)
            if cr_form == "substituted":
                part.loc[part["sample"].eq("Cr-LNCM"), ["Formula", "canonical_key", "in_training"]] = \
                    [PANG_CR_SUBSTITUTED, sub_key, sub_key in train_keys]
        X = l8.features(evaluator, part["Formula"].tolist(), rate, cycle)
        full = np.asarray(evaluator.model.predict(X, thread_count=1), dtype=float)
        boot = np.column_stack([np.asarray(m.predict(X, thread_count=1), dtype=float) for m in models])
        for i, r in part.iterrows():
            pred_rows.append({"paper": paper, "scenario": label, "sample": r["sample"], "dopant": r["dopant"],
                              "Formula": r["Formula"], "C_rate": rate, "cycle": cycle, "measured": r[column],
                              "pred_full": full[i], "pred_full_clipped": float(np.clip(full[i], *evaluator.prediction_clip)),
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
                "paper": paper, "scenario": label, "doped": r["sample"], "dopant": r["dopant"],
                "doped_formula": r["Formula"], "baseline_formula": part.loc[b, "Formula"],
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
    preds.to_csv(OUT / "08b_predictions.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(OUT / "08b_pairs.csv", index=False, encoding="utf-8-sig")

    overlap = doi_overlap(list(dict.fromkeys(samples["DOI"])))
    overlap.to_csv(OUT / "08b_doi_overlap.csv", index=False, encoding="utf-8-sig")

    # check that the 08 rows reproduce outputs/08_pairs.csv exactly
    old = pd.read_csv(OUT / "08_pairs.csv", encoding="utf-8-sig")
    new = pairs[pairs["paper"].isin(old["paper"].unique())].reset_index(drop=True)
    cols = ["measured_delta", "pred_delta_full", "boot_delta_P05", "boot_delta_P95"]
    same_08 = bool(len(old) == len(new) and np.allclose(old[cols].to_numpy(float), new[cols].to_numpy(float),
                                                        equal_nan=True, rtol=0, atol=1e-9))
    write_json("08b_summary.json", {"n_samples": len(samples), "n_in_training": int(samples["in_training"].sum()),
                                    "in_training": samples[samples["in_training"]][["paper", "sample", "Formula"]]
                                    .to_dict(orient="records"),
                                    "pang_cr_substituted_in_training": sub_key in train_keys,
                                    "v1_pairs_identical_to_08": same_08,
                                    "prediction_clip": evaluator.prediction_clip})
    log(f"08 pairs reproduced: {same_08}")


if __name__ == "__main__":
    main()
