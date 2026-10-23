"""Strict cationic-redox budget check on saved artifacts only.

Reads the saved release outputs (positive controls, 20 reference TopK tables,
clean data) and, for ten named compositions, predicts with the saved CatBoost
model.  Nothing is trained, no search is run, and no release code is changed.

Strict budget definition:
  - upper oxidation states: Mn 4, Ni 4, Co 4, Fe 3, Cr 6, V 5, Cu 3, Ti 4;
  - every other cation keeps one fixed valence (FIXED_VALENCE below);
    Li +1, O -2, F -1;
  - removable electrons = sum(x_TM * upper_TM) - (total valence the redox TMs
    must carry for charge neutrality), capped at the Li content, floored at 0;
  - mAh/g = electrons * F / (3.6 * formula weight), as in the original G4.
Outputs go to outputs/strict_redox_*.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition

HERE = Path(__file__).resolve().parent
SAVED = HERE.parents[1] / "paper_release_v1"
WORK = HERE.parents[1] / "paper_release_v1"
OUT = HERE.parent / "outputs"
SEEDS = list(range(42, 62))
FARADAY = 96485.0

REDOX_UPPER = {"Mn": 4, "Ni": 4, "Co": 4, "Fe": 3, "Cr": 6, "V": 5, "Cu": 3, "Ti": 4}
# Fixed valences for every other element that occurs in the controls, the 20 reference
# TopK tables and the ten named compositions (pymatgen common_oxidation_states; for Mo and P,
# which have several common states, the molybdate/phosphate value is used).
FIXED_VALENCE = {
    "Li": 1, "Na": 1, "K": 1, "Mg": 2, "Zn": 2, "Al": 3, "Ga": 3, "La": 3, "Nd": 3,
    "Sm": 3, "Gd": 3, "Zr": 4, "Nb": 5, "Mo": 6, "P": 5, "O": -2, "F": -1,
}
ANIONS = {"O", "F"}
TEN = [
    "Li2.1Mn0.7Cr0.3O2.865F0.135", "Li2.2Mn0.7Cr0.3O2.865F0.135", "Li2.15Mn0.7Cr0.3O2.865F0.135",
    "Li2.2Mn0.72Cr0.28O2.865F0.135", "Li1.95Mn0.72Cr0.28O2.865F0.135", "Li1.95Mn0.7Cr0.3O2.865F0.135",
    "Li1.9Mn0.7Cr0.3O2.865F0.135", "Li2MnO3", "Li2Mn0.7Cr0.3O3", "Li2MnO2.865F0.135",
]


def strict_budget(formula: str) -> dict[str, float]:
    comp = Composition(formula)
    amounts = comp.get_el_amt_dict()
    unknown = [e for e in amounts if e not in REDOX_UPPER and e not in FIXED_VALENCE]
    if unknown:
        raise KeyError(f"No valence rule for {unknown} in {formula}")
    fixed_charge = sum(x * FIXED_VALENCE[e] for e, x in amounts.items() if e in FIXED_VALENCE)
    tm_required = -fixed_charge  # valence the redox TMs must carry for neutrality
    tm_upper = sum(x * REDOX_UPPER[e] for e, x in amounts.items() if e in REDOX_UPPER)
    lithium = float(amounts.get("Li", 0.0))
    raw_electrons = tm_upper - tm_required
    electrons = max(0.0, min(lithium, raw_electrons))
    weight = float(comp.weight)
    cations = sum(x for e, x in amounts.items() if e not in ANIONS)
    anions = sum(x for e, x in amounts.items() if e in ANIONS)
    return {
        "Li_per_fu": lithium,
        "tm_required_valence": tm_required,
        "tm_upper_valence": tm_upper,
        "removable_electrons_before_cap": raw_electrons,
        "strict_electrons": electrons,
        "formula_weight": weight,
        "strict_budget_mAh_g": electrons * FARADAY / (3.6 * weight),
        "cation_to_anion_ratio": cations / anions if anions else np.nan,
    }


def controls() -> pd.DataFrame:
    known = pd.read_csv(SAVED / "results" / "known_control_screened.csv")
    clean = pd.read_csv(SAVED / "data" / "clean_rows.csv")
    measured = clean.groupby("canonical_key")["Value"].agg(measured_median="median", measured_max="max",
                                                         n_records="size")
    rows = []
    for row in known.itertuples(index=False):
        strict = strict_budget(row.Formula)
        m = measured.loc[row.canonical_key]
        rows.append({
            "proto_name": row.proto_name, "Formula": row.Formula, "canonical_key": row.canonical_key,
            "original_G4_budget_mAh_g": float(row.q_cationic_redox),
            **strict,
            "predicted_y_pred": float(row.y_pred),
            "measured_median": float(m.measured_median), "measured_max": float(m.measured_max),
            "n_records": int(m.n_records),
            "median_exceeds_strict": bool(m.measured_median > strict["strict_budget_mAh_g"]),
            "max_exceeds_strict": bool(m.measured_max > strict["strict_budget_mAh_g"]),
            "prediction_exceeds_strict": bool(row.y_pred > strict["strict_budget_mAh_g"]),
            "original_g4_pass": bool(row.g4_pass),
        })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "04_strict_redox_controls.csv", index=False)
    return table


def ten_compositions() -> pd.DataFrame:
    sys.path.insert(0, str(WORK / "src"))
    import search_screen_benchmark as ssb  # noqa: E402

    evaluator = ssb.CandidateEvaluator()
    candidates = []
    for formula in TEN:
        comp = Composition(formula)
        candidates.append({"comp": comp.get_el_amt_dict(), "Formula": comp.formula.replace(" ", ""),
                           "canonical_key": ssb.canonical_key(comp), "proto_name": "Li2MnO3", "is_doped": 1})
    evaluated = evaluator.evaluate(candidates)
    rows = []
    for formula, pred in zip(TEN, evaluated["y_pred"]):
        strict = strict_budget(formula)
        amounts = Composition(formula).get_el_amt_dict()
        mn, cr = amounts.get("Mn", 0.0), amounts.get("Cr", 0.0)
        required = strict["tm_required_valence"]
        _, original_budget, _ = ssb.redox_budget(formula)
        rows.append({
            "Formula": formula, **strict,
            "original_G4_budget_mAh_g": original_budget,
            "predicted_y_pred": float(pred),
            "prediction_exceeds_strict": bool(pred > strict["strict_budget_mAh_g"]),
            "Cr_avg_valence_if_Mn4": (required - 4 * mn) / cr if cr else np.nan,
            "Mn_avg_valence_if_Cr3": (required - 3 * cr) / mn if mn else np.nan,
        })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "04_strict_redox_ten_compositions.csv", index=False)
    return table


def recurrence(p95: float) -> dict[str, object]:
    budgets: dict[str, float] = {}
    frames = []
    for seed in SEEDS:
        frame = pd.read_csv(SAVED / "results" / "search_candidates" / f"nsga2__floor_on_seed{seed}_topk_screened.csv")
        frame["seed"] = seed
        for formula in frame["Formula"].unique():
            if formula not in budgets:
                budgets[formula] = strict_budget(formula)["strict_budget_mAh_g"]
        frame["strict_budget_mAh_g"] = frame["Formula"].map(budgets)
        frame["g4_strict_pass"] = pd.to_numeric(frame["y_pred"]).le(frame["strict_budget_mAh_g"])
        radius = pd.to_numeric(frame["radius_mismatch"], errors="coerce")
        g12 = frame["g1_pass"].astype(bool) & frame["g2_pass"].astype(bool)
        frame["p95_final_original"] = g12 & frame["g4_pass"].astype(bool) & radius.le(p95)
        frame["p95_final_strict"] = g12 & frame["g4_strict_pass"] & radius.le(p95)
        frames.append(frame)
    rows = pd.concat(frames, ignore_index=True)
    records = []
    for key, part in rows.groupby("canonical_key", sort=True):
        seeds = sorted(part["seed"].unique())
        strict_seeds = sorted(part.loc[part["p95_final_strict"], "seed"].unique())
        records.append({
            "canonical_key": key, "Formula": part["Formula"].iloc[0], "proto_name": part["proto_name"].iloc[0],
            "topk_seed_count": len(seeds),
            "strict_p95_final_seed_count": len(strict_seeds),
            "strict_budget_mAh_g": float(part["strict_budget_mAh_g"].iloc[0]),
            "y_pred_mean": float(part["y_pred"].mean()),
            "consensus_strict": (len(seeds) >= 10 and len(strict_seeds) == len(seeds)
                                 and bool(part["g1_pass"].astype(bool).all())
                                 and bool(part["g2_pass"].astype(bool).all())
                                 and bool(part["g4_strict_pass"].all())),
        })
    union = pd.DataFrame(records)
    union.to_csv(OUT / "04_strict_redox_recurrence_union.csv", index=False)
    original = pd.read_csv(SAVED / "results" / "consensus_candidates.csv")
    strict_consensus = union[union["consensus_strict"]].sort_values("topk_seed_count", ascending=False)
    per_seed = rows.groupby("seed")[["p95_final_original", "p95_final_strict"]].sum().astype(int)
    return {
        "union": int(len(union)),
        "original_consensus": sorted(original["Formula"]),
        "strict_consensus": strict_consensus[["Formula", "proto_name", "topk_seed_count", "strict_budget_mAh_g",
                                              "y_pred_mean"]].to_dict(orient="records"),
        "original_retained": sorted(set(original["Formula"]) & set(strict_consensus["Formula"])),
        "original_dropped": sorted(set(original["Formula"]) - set(strict_consensus["Formula"])),
        "new_recurrent": sorted(set(strict_consensus["Formula"]) - set(original["Formula"])),
        "candidates_with_10plus_seeds": int((union["topk_seed_count"] >= 10).sum()),
        "per_seed_p95_survivors": per_seed.reset_index().to_dict(orient="records"),
        "per_seed_mean_original": float(per_seed["p95_final_original"].mean()),
        "per_seed_mean_strict": float(per_seed["p95_final_strict"].mean()),
    }


def main() -> None:
    p95 = float(json.loads((SAVED / "results" / "known_control_summary.json").read_text(encoding="utf-8"))[
        "positive_control_radius_p95_after_g1_g2"])
    control_table = controls()
    ten = ten_compositions()
    rec = recurrence(p95)
    by_proto = control_table.groupby("proto_name").agg(
        n=("Formula", "size"),
        median_exceeds_strict=("median_exceeds_strict", "sum"),
        max_exceeds_strict=("max_exceeds_strict", "sum"),
        prediction_exceeds_strict=("prediction_exceeds_strict", "sum"),
        original_g4_fail=("original_g4_pass", lambda s: int((~s).sum())),
    ).reset_index()
    by_proto.to_csv(OUT / "04_strict_redox_controls_by_prototype.csv", index=False)
    summary = {
        "redox_upper": REDOX_UPPER, "fixed_valence": FIXED_VALENCE, "p95": p95,
        "controls_by_prototype": by_proto.to_dict(orient="records"),
        "controls_totals": {k: int(by_proto[k].sum()) for k in
                            ["n", "median_exceeds_strict", "max_exceeds_strict", "prediction_exceeds_strict",
                             "original_g4_fail"]},
        "recurrence": rec,
    }
    (OUT / "04_strict_redox_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(by_proto.to_string())
        print(ten.round(3).to_string())
    print(json.dumps({k: v for k, v in rec.items() if k != "per_seed_p95_survivors"}, indent=1, default=float))


if __name__ == "__main__":
    main()
