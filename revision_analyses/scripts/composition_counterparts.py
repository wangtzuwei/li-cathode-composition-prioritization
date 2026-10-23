"""Counterpart compositions for the seven Table 3 candidates.

Uses only saved artifacts and the 100 bootstrap models saved on 2026-10-22
(models_local/bootstrap_models/, written by train_bootstrap_models.py).  No search, no training, no change to
existing programs.

Step 1 checks that the 100 saved bootstrap models reproduce the Table 3
P05-P95 (rounded to 0.1 mAh/g, as printed in the manuscript); the script stops
if any value differs.
Step 2 builds, for each candidate Li_x Mn_a Cr_b O2.865 F0.135 (Li kept):
  (a) no F, O = 3.0         -> Li_x Mn_a Cr_b O3
  (b) no Cr, Mn = 1.0       -> Li_x Mn1 O2.865 F0.135
  (c) no F and no Cr        -> Li_x Mn1 O3
and evaluates the 7 candidates, 21 counterparts and Li2MnO3.
Outputs: outputs/05_counterpart_*.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from pymatgen.core import Composition

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1] / "paper_release_v1"
SAVED = HERE.parents[1] / "paper_release_v1"
MODELS = HERE.parent / "models_local" / "bootstrap_models"
OUT = HERE.parent / "outputs"
sys.path.insert(0, str(WORK / "src"))
sys.path.insert(0, str(HERE))

import search_screen_benchmark as ssb  # noqa: E402
from strict_redox_budget import strict_budget  # noqa: E402

TABLE3 = [  # formula, manuscript P05, manuscript P95
    ("Li2.1Mn0.7Cr0.3O2.865F0.135", 178.4, 230.9),
    ("Li2.2Mn0.7Cr0.3O2.865F0.135", 171.8, 229.2),
    ("Li2.15Mn0.7Cr0.3O2.865F0.135", 176.6, 232.0),
    ("Li2.2Mn0.72Cr0.28O2.865F0.135", 173.2, 229.9),
    ("Li1.95Mn0.72Cr0.28O2.865F0.135", 181.0, 228.8),
    ("Li1.95Mn0.7Cr0.3O2.865F0.135", 180.8, 229.4),
    ("Li1.9Mn0.7Cr0.3O2.865F0.135", 176.4, 228.2),
]
PROTO = "Li2MnO3"


def num(x: float) -> str:
    return f"{x:g}"


def counterparts(formula: str) -> list[tuple[str, str]]:
    amounts = Composition(formula).get_el_amt_dict()
    li, mn, cr = num(amounts["Li"]), num(amounts["Mn"]), num(amounts["Cr"])
    return [
        ("(a) no F", f"Li{li}Mn{mn}Cr{cr}O3"),
        ("(b) no Cr", f"Li{li}Mn1O2.865F0.135"),
        ("(c) no F, no Cr", f"Li{li}Mn1O3"),
    ]


def load_models() -> tuple[list[CatBoostRegressor], list[str]]:
    manifest = json.loads((MODELS / "manifest.json").read_text(encoding="utf-8"))
    models = []
    for item in manifest["models"]:
        model = CatBoostRegressor()
        model.load_model(str(MODELS / item["file"]))
        models.append(model)
    return models, manifest["feature_columns"]


def main() -> None:
    evaluator = ssb.CandidateEvaluator()
    models, feature_columns = load_models()
    if feature_columns != list(evaluator.feature_columns):
        raise RuntimeError("Bootstrap model feature order differs from the saved feature table")

    def features(formulas: list[str], rate: float, cycle: float) -> pd.DataFrame:
        rows = []
        for formula in formulas:
            comp = Composition(formula)
            rows.append(evaluator._composition_row({"comp": comp.get_el_amt_dict(), "Formula": formula,
                                                     "canonical_key": ssb.canonical_key(comp)}))
        X = pd.DataFrame(rows).reindex(columns=evaluator.comp_columns)
        X["temperature_K"] = 298.15
        X["log1p_current_rate_C"] = math.log1p(rate)
        X["log1p_cycle_number"] = math.log1p(cycle)
        X["log1p_current_rate_x_cycle"] = math.log1p(rate * cycle)
        return X.reindex(columns=evaluator.feature_columns)

    def bootstrap(formulas: list[str]) -> np.ndarray:
        X = features(formulas, 0.5, 1.0)
        return np.column_stack([np.asarray(m.predict(X, thread_count=1), dtype=float) for m in models])

    # Step 1: Table 3 P05-P95 from the saved 100 models must equal the manuscript.
    t3 = [f for f, _, _ in TABLE3]
    pred = bootstrap(t3)
    check = []
    for (formula, p05_ms, p95_ms), row in zip(TABLE3, pred):
        p05, p95 = np.percentile(row, 5), np.percentile(row, 95)
        check.append({"Formula": formula, "P05_manuscript": p05_ms, "P05_models": round(float(p05), 1),
                      "P95_manuscript": p95_ms, "P95_models": round(float(p95), 1),
                      "match": round(float(p05), 1) == p05_ms and round(float(p95), 1) == p95_ms,
                      "P05_raw": float(p05), "P95_raw": float(p95)})
    check_df = pd.DataFrame(check)
    check_df.to_csv(OUT / "05_table3_interval_check.csv", index=False)
    print(check_df.to_string())
    if not check_df["match"].all():
        print("STOP: Table 3 P05-P95 from the saved bootstrap models differ from the manuscript.")
        sys.exit(2)

    # Step 2: the 29 compositions.
    entries = []
    for formula, _, _ in TABLE3:
        entries.append((formula, "candidate", formula))
        for label, other in counterparts(formula):
            entries.append((formula, label, other))
    entries.append(("Li2MnO3", "reference", "Li2MnO3"))
    formulas = [Composition(e[2]).formula.replace(" ", "") for e in entries]

    candidates = []
    for formula in formulas:
        comp = Composition(formula)
        candidates.append({"comp": comp.get_el_amt_dict(), "Formula": formula,
                           "canonical_key": ssb.canonical_key(comp), "proto_name": PROTO, "is_doped": 1})
    evaluated = evaluator.evaluate(candidates)
    boot = bootstrap(formulas)
    contexts = []
    for rate in [0.1, 0.5, 1.0, 5.0]:
        for cycle in [1.0, 50.0, 100.0, 200.0]:
            contexts.append(np.asarray(evaluator.model.predict(features(formulas, rate, cycle), thread_count=1), dtype=float))
    context_min = np.min(np.column_stack(contexts), axis=1)

    fractions = []
    for formula in formulas:
        amounts = Composition(formula).fractional_composition.get_el_amt_dict()
        fractions.append([amounts.get(e, 0.0) for e in ssb.PERIODIC_ELEMENTS])
    distance, index = evaluator.comp_nn.kneighbors(np.asarray(fractions), n_neighbors=ssb.KNN_K)
    train_keys = list(evaluator.unique_train.index)
    clean = pd.read_csv(SAVED / "data" / "clean_rows.csv")
    key_to_formula = clean.groupby("canonical_key")["canonical_formula"].first()

    p95 = float(json.loads((SAVED / "results" / "known_control_summary.json").read_text(encoding="utf-8"))[
        "positive_control_radius_p95_after_g1_g2"])
    rows = []
    for i, ((parent, label, _), formula) in enumerate(zip(entries, formulas)):
        g1, _, _ = ssb.gate1_ionicity(formula)
        g2, ratio = ssb.gate2_li_tm(formula, PROTO)
        rms, _ = ssb.radius_mismatch_metrics(formula, PROTO)
        _, q_cationic, _ = ssb.redox_budget(formula)
        strict = strict_budget(formula)
        y_pred = float(evaluated["y_pred"].iloc[i])
        rows.append({
            "parent_candidate": parent, "variant": label, "Formula": formula,
            "y_pred_saved_model": y_pred,
            "bootstrap_P05": float(np.percentile(boot[i], 5)), "bootstrap_P95": float(np.percentile(boot[i], 95)),
            "context_min_16": float(context_min[i]),
            "G1_pass": bool(g1), "G2_pass": bool(g2), "Li_TM_ratio": ratio,
            "G3_weighted_rms": rms, "G3_pass_P95": bool(rms <= p95),
            "G4_original_budget": q_cationic, "G4_original_pass": bool(y_pred <= q_cationic),
            "G4_strict_budget": strict["strict_budget_mAh_g"],
            "G4_strict_pass": bool(y_pred <= strict["strict_budget_mAh_g"]),
            "cation_to_anion_ratio": strict["cation_to_anion_ratio"],
            "comp_novelty_mean_5nn": float(distance[i].mean()),
            "nearest_5_distances": ";".join(f"{d:.4f}" for d in distance[i]),
            "nearest_5_training_formulas": ";".join(str(key_to_formula.get(train_keys[j], train_keys[j])) for j in index[i]),
            "model_distance": float(evaluated["model_distance"].iloc[i]),
            "comp_novelty_evaluator": float(evaluated["comp_novelty"].iloc[i]),
        })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "05_counterpart_controls.csv", index=False)
    with pd.option_context("display.width", 260, "display.max_columns", 30):
        print(table.drop(columns=["nearest_5_distances", "nearest_5_training_formulas"]).round(3).to_string())


if __name__ == "__main__":
    main()
