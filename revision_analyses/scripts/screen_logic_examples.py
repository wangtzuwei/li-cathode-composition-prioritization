"""Analysis 03: evaluate G1-G4 for ten compositions with the release's own functions.

Calls search_screen_benchmark.gate1_ionicity, gate2_li_tm, radius_mismatch_metrics,
redox_budget and CandidateEvaluator.evaluate (saved model, prediction only) from
paper_release_v1/src.  Nothing is trained and no search is run.
Output: outputs/03_screen_logic_examples.csv (+ .json with per-element detail).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from pymatgen.core import Composition

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1] / "paper_release_v1"
SAVED = HERE.parents[1] / "paper_release_v1"
OUT = HERE.parent / "outputs"
sys.path.insert(0, str(WORK / "src"))

import search_screen_benchmark as ssb  # noqa: E402

FORMULAS = [
    "Li2.1Mn0.7Cr0.3O2.865F0.135", "Li2.2Mn0.7Cr0.3O2.865F0.135", "Li2.15Mn0.7Cr0.3O2.865F0.135",
    "Li2.2Mn0.72Cr0.28O2.865F0.135", "Li1.95Mn0.72Cr0.28O2.865F0.135", "Li1.95Mn0.7Cr0.3O2.865F0.135",
    "Li1.9Mn0.7Cr0.3O2.865F0.135", "Li2MnO3", "Li2Mn0.7Cr0.3O3", "Li2MnO2.865F0.135",
]
PROTO = "Li2MnO3"


def main() -> None:
    thresholds = json.loads((SAVED / "results" / "known_control_summary.json").read_text(encoding="utf-8"))
    p95 = float(thresholds["positive_control_radius_p95_after_g1_g2"])
    consensus = pd.read_csv(SAVED / "results" / "consensus_candidates.csv").set_index("Formula")
    evaluator = ssb.CandidateEvaluator()
    candidates = []
    for formula in FORMULAS:
        comp = Composition(formula)
        candidates.append({"comp": comp.get_el_amt_dict(), "Formula": comp.formula.replace(" ", ""),
                           "canonical_key": ssb.canonical_key(comp), "proto_name": PROTO, "is_doped": 1})
    evaluated = evaluator.evaluate(candidates)

    rows, detail = [], {}
    for formula, row in zip(FORMULAS, evaluated.itertuples(index=False)):
        comp = Composition(formula).get_el_amt_dict()
        allowed = {}
        for element in sorted(comp):
            states = ssb.oxidation_states(element)
            allowed[element] = ([s for s in states if s < 0] if element in ssb.ANIONS else [s for s in states if s > 0])
        charge_min = sum(comp[e] * min(allowed[e]) for e in sorted(comp))
        charge_max = sum(comp[e] * max(allowed[e]) for e in sorted(comp))
        g1_pass, min_abs_charge, pauling = ssb.gate1_ionicity(row.Formula)
        g2_pass, ratio = ssb.gate2_li_tm(row.Formula, PROTO)
        rms, max_rel = ssb.radius_mismatch_metrics(row.Formula, PROTO)
        q_theoretical, q_cationic, redox_e = ssb.redox_budget(row.Formula)
        lithium = float(comp.get("Li", 0.0))
        y_pred = float(row.y_pred)
        saved_pool = float(consensus.loc[formula, "y_pred_mean_across_occurrences"]) if formula in consensus.index else None
        rows.append({
            "Formula": formula, "Li_per_fu": lithium,
            "G1_charge_interval_min": charge_min, "G1_charge_interval_max": charge_max,
            "G1_min_abs_charge": min_abs_charge, "G1_pauling_pass": pauling, "G1_pass": g1_pass,
            "G2_Li_TM_ratio": ratio, "G2_pass": g2_pass,
            "G3_weighted_rms": rms, "G3_max_diagnostic": max_rel, "G3_pass_P95": bool(rms <= p95),
            "G4_redox_electrons_per_fu": redox_e, "G4_min_Li_redox_e": min(lithium, redox_e),
            "G4_Q_Li_theoretical_mAh_g": q_theoretical, "G4_Q_cationic_redox_mAh_g": q_cationic,
            "formula_weight_g_mol": float(Composition(formula).weight),
            "y_pred_unclipped": float(row.y_pred_unclipped), "y_pred": y_pred,
            "y_pred_saved_pool_mean": saved_pool, "y_discounted": float(row.y_discounted),
            "model_distance": float(row.model_distance),
            "G4_pass": bool(y_pred <= q_cationic),
            "all_G1_G4_pass_P95": bool(g1_pass and g2_pass and rms <= p95 and y_pred <= q_cationic),
        })
        detail[formula] = {"amounts": comp, "allowed_states_used_by_G1": allowed}
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "03_screen_logic_examples.csv", index=False)
    (OUT / "03_screen_logic_examples.json").write_text(
        json.dumps({"p95_threshold": p95, "prediction_clip": evaluator.prediction_clip, "detail": detail,
                    "rows": rows}, indent=2), encoding="utf-8")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(table.round(4).to_string())
    print("prediction clip", evaluator.prediction_clip, "P95", p95)
    print(json.dumps({k: v["allowed_states_used_by_G1"] for k, v in list(detail.items())[:1]}))


if __name__ == "__main__":
    main()
