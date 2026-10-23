#!/usr/bin/env python3
"""Assemble final P90/P95 portfolios, 20-seed stability, and manuscript JSON."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from postprocess_release import make_li_tm_window_sensitivity, make_literature_context_table


ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "paper_release_v1"
RESULTS = RELEASE / "results"
SEARCH = RESULTS / "search_candidates"
SEEDS = list(range(42, 62))
REFERENCE_SEED = 42


def jaccard(sets: list[set[str]]) -> float:
    values = []
    for left in range(len(sets)):
        for right in range(left + 1, len(sets)):
            union = sets[left] | sets[right]
            values.append(len(sets[left] & sets[right]) / len(union) if union else np.nan)
    return float(np.nanmean(values))


def reference_arm() -> str:
    decision = json.loads((RESULTS / "nsga_decision.json").read_text(encoding="utf-8"))
    return "nsga2__floor_on" if decision["retain_nsga2_as_substantive_performance_claim"] else "random__floor_on"


def thresholds() -> tuple[float, float, dict[str, object]]:
    known = json.loads((RESULTS / "known_control_summary.json").read_text(encoding="utf-8"))
    return (
        float(known["positive_control_radius_p90_after_g1_g2"]),
        float(known["positive_control_radius_p95_after_g1_g2"]),
        known,
    )


def screen_reference(arm: str, p90: float, p95: float) -> dict[str, object]:
    frame = pd.read_csv(SEARCH / f"{arm}_seed{REFERENCE_SEED}_topk_screened.csv").copy()
    base = frame["g1_pass"].astype(bool) & frame["g2_pass"].astype(bool) & frame["g4_pass"].astype(bool)
    radius = pd.to_numeric(frame["radius_mismatch"], errors="coerce")
    frame["pass_conservative_p90_final"] = base & radius.le(p90)
    frame["pass_reference_p95_final"] = base & radius.le(p95)
    frame["risk_tier"] = np.select(
        [
            frame["pass_conservative_p90_final"],
            base & radius.gt(p90) & radius.le(p95),
            base & radius.gt(p95),
            frame["g1_pass"].astype(bool) & ~frame["g2_pass"].astype(bool),
            ~frame["g1_pass"].astype(bool),
            frame["g1_pass"].astype(bool) & frame["g2_pass"].astype(bool) & ~frame["g4_pass"].astype(bool),
        ],
        [
            "A_conservative_P90_core",
            "B_reference_P95_extension",
            "C_beyond_P95_radius_risk",
            "D_Li_TM_window_risk",
            "E_formal_ionicity_risk",
            "F_redox_budget_risk",
        ],
        default="Unclassified",
    )
    frame.to_csv(RESULTS / "primary_reference_topk_risk_tiers.csv", index=False)
    core = frame[frame["pass_conservative_p90_final"]].copy()
    portfolio = frame[frame["pass_reference_p95_final"]].copy()
    core.to_csv(RESULTS / "primary_reference_P90_core.csv", index=False)
    core.to_csv(RESULTS / "primary_reference_strict_shortlist.csv", index=False)
    portfolio.to_csv(RESULTS / "primary_reference_P95_portfolio.csv", index=False)
    portfolio.to_csv(RESULTS / "primary_reference_control_anchored_portfolio.csv", index=False)
    return {
        "reference_arm": arm,
        "reference_seed": REFERENCE_SEED,
        "reference_topk": int(len(frame)),
        "reference_p90_core_survivors": int(len(core)),
        "reference_p95_portfolio_survivors": int(len(portfolio)),
        "reference_p95_extension": int(len(portfolio) - len(core)),
        "reference_p90_core_prototype_coverage": int(core["proto_name"].nunique()),
        "reference_p95_portfolio_prototype_coverage": int(portfolio["proto_name"].nunique()),
        "risk_tier_counts": {str(key): int(value) for key, value in frame["risk_tier"].value_counts().items()},
    }


def stability_summary(arm: str, p90: float, p95: float) -> dict[str, object]:
    frames = []
    per_seed = []
    for seed in SEEDS:
        frame = pd.read_csv(SEARCH / f"{arm}_seed{seed}_topk_screened.csv").copy()
        base = frame["g1_pass"].astype(bool) & frame["g2_pass"].astype(bool) & frame["g4_pass"].astype(bool)
        radius = pd.to_numeric(frame["radius_mismatch"], errors="coerce")
        frame["conservative_p90"] = base & radius.le(p90)
        frame["reference_p95"] = base & radius.le(p95)
        frames.append(frame)
        per_seed.append(
            {
                "seed": seed,
                "topk": int(len(frame)),
                "conservative_p90": int(frame["conservative_p90"].sum()),
                "reference_p95": int(frame["reference_p95"].sum()),
            }
        )
    topk_sets = [set(frame["canonical_key"].astype(str)) for frame in frames]
    p90_sets = [set(frame.loc[frame["conservative_p90"], "canonical_key"].astype(str)) for frame in frames]
    p95_sets = [set(frame.loc[frame["reference_p95"], "canonical_key"].astype(str)) for frame in frames]
    recurrence = pd.read_csv(RESULTS / "candidate_recurrence_union.csv")
    output = {
        "primary_arm": arm,
        "seeds": SEEDS,
        "consensus_definition": "canonical candidate appears in at least 10 of 20 search seeds",
        "per_seed_counts": per_seed,
        "topk_union": int(len(set().union(*topk_sets))),
        "topk_all_twenty": int(recurrence["topk_seed_count"].eq(20).sum()),
        "topk_consensus_10_of_20": int(recurrence["topk_seed_count"].ge(10).sum()),
        "topk_mean_pairwise_jaccard": jaccard(topk_sets),
        "conservative_p90_union": int(len(set().union(*p90_sets))),
        "conservative_p90_all_twenty": int(recurrence["p90_final_seed_count"].eq(20).sum()),
        "conservative_p90_consensus_10_of_20": int(
            (recurrence["topk_seed_count"].ge(10) & recurrence["passes_p90_every_occurrence"]).sum()
        ),
        "conservative_p90_mean_pairwise_jaccard": jaccard(p90_sets),
        "reference_p95_union": int(len(set().union(*p95_sets))),
        "reference_p95_all_twenty": int(recurrence["p95_final_seed_count"].eq(20).sum()),
        "reference_p95_consensus_10_of_20": int(
            (recurrence["topk_seed_count"].ge(10) & recurrence["passes_p95_every_occurrence"]).sum()
        ),
        "reference_p95_mean_pairwise_jaccard": jaccard(p95_sets),
        "primary_scientific_shortlist_n": int(recurrence["primary_consensus_candidate"].sum()),
    }
    recurrence.to_csv(RESULTS / "primary_candidate_stability.csv", index=False)
    (RESULTS / "candidate_stability_summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def positive_control_detail(p90: float, p95: float, known_summary: dict[str, object]) -> dict[str, object]:
    known = pd.read_csv(RESULTS / "known_control_screened.csv").copy()
    reached = known["g1_pass"].astype(bool) & known["g2_pass"].astype(bool)
    radius = pd.to_numeric(known["radius_mismatch"], errors="coerce")
    g4 = known["g4_pass"].astype(bool)
    known["g3_p90_pass"] = radius.le(p90)
    known["g3_p95_pass"] = radius.le(p95)
    known["first_failure_p90"] = np.select(
        [
            ~known["g1_pass"].astype(bool),
            known["g1_pass"].astype(bool) & ~known["g2_pass"].astype(bool),
            reached & ~known["g3_p90_pass"],
            reached & known["g3_p90_pass"] & ~g4,
        ],
        ["G1", "G2", "G3", "G4"],
        default="Pass",
    )
    known.to_csv(RESULTS / "known_control_screened.csv", index=False)
    rows = []
    for proto, part in known.groupby("proto_name", sort=True):
        part_reached = part[part["g1_pass"].astype(bool) & part["g2_pass"].astype(bool)]
        rows.append(
            {
                "proto_name": proto,
                "applicable_n": int(len(part)),
                "g1_survivors_n": int(part["g1_pass"].sum()),
                "g1_g2_survivors_n": int(part["pass_g1_g2"].sum()),
                "g3_p90_survivors_n": int((part["pass_g1_g2"].astype(bool) & part["g3_p90_pass"]).sum()),
                "g3_p90_losses_after_g1_g2_n": int((~part_reached["g3_p90_pass"]).sum()),
                "g3_p95_survivors_n": int((part["pass_g1_g2"].astype(bool) & part["g3_p95_pass"]).sum()),
                "g3_p95_losses_after_g1_g2_n": int((~part_reached["g3_p95_pass"]).sum()),
                "final_p90_survivors_n": int((part["pass_g1_g2"].astype(bool) & part["g3_p90_pass"] & part["g4_pass"].astype(bool)).sum()),
                "final_p95_survivors_n": int((part["pass_g1_g2"].astype(bool) & part["g3_p95_pass"] & part["g4_pass"].astype(bool)).sum()),
            }
        )
    pd.DataFrame(rows).to_csv(RESULTS / "positive_control_prototype_summary.csv", index=False)
    policy = known_summary["cross_fitted_policy_validation"]
    summary = {
        "g1_g2_denominator": int(reached.sum()),
        "p90_threshold": p90,
        "p95_threshold": p95,
        "p90_g3_survivors_full_sample": int((reached & radius.le(p90)).sum()),
        "p95_g3_survivors_full_sample": int((reached & radius.le(p95)).sum()),
        "p90_final_survivors_full_sample": int((reached & radius.le(p90) & g4).sum()),
        "p95_final_survivors_full_sample": int((reached & radius.le(p95) & g4).sum()),
        "p90_leave_one_control_out": policy["policies"]["P90"],
        "p95_leave_one_control_out": policy["policies"]["P95"],
    }
    (RESULTS / "positive_control_dual_g3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def update_results(
    reference: dict[str, object],
    stability: dict[str, object],
    controls: dict[str, object],
    arm: str,
    p90: float,
    p95: float,
) -> dict[str, object]:
    recurrence = json.loads((RESULTS / "candidate_recurrence_summary.json").read_text(encoding="utf-8"))
    uncertainty = json.loads((RESULTS / "candidate_uncertainty_summary.json").read_text(encoding="utf-8"))
    extended = json.loads((RESULTS / "extended_validation_summary.json").read_text(encoding="utf-8"))
    search = pd.read_csv(RESULTS / "search_seed_summary.csv")
    selected = search[search["arm"].eq(arm)]
    primary = {
        **reference,
        "primary_arm": arm,
        "selection_rule": "scientific shortlist requires recurrence in at least 10 of 20 seeds and P95-final pass in every occurrence; P90 marks a nested conservative core and seed 42 is retained only as an auditable example release",
        "reason": "The canonical elitist NSGA-II comparison is interpreted from the 20-seed paired feasible-hypervolume result; the prototype floor is a separate design constraint.",
        "positive_control_p90": p90,
        "positive_control_p95": p95,
        "twenty_seed_primary_survivors_mean": float(selected["g4_survivors"].mean()),
        "twenty_seed_primary_survivors_sd": float(selected["g4_survivors"].std(ddof=1)),
        "twenty_seed_raw_prediction_mean": float(selected["y_pred_mean"].mean()),
        "twenty_seed_raw_prediction_sd": float(selected["y_pred_mean"].std(ddof=1)),
        "candidate_recurrence": recurrence,
        "primary_scientific_shortlist_n": recurrence["primary_consensus_candidates"],
    }
    (RESULTS / "primary_analysis_summary.json").write_text(json.dumps(primary, indent=2), encoding="utf-8")

    consensus = pd.read_csv(RESULTS / "consensus_candidates.csv")
    uncertainty_frame = pd.read_csv(RESULTS / "candidate_model_uncertainty.csv")
    context_frame = pd.read_csv(RESULTS / "candidate_context_robustness.csv")
    examples = consensus.merge(
        uncertainty_frame.drop(columns=["Formula", "proto_name", "topk_seed_count", "primary_consensus_candidate", "radius_mismatch"], errors="ignore"),
        on="canonical_key",
        how="left",
    ).merge(
        context_frame[["canonical_key", "context_prediction_min", "context_prediction_max", "context_prediction_range"]],
        on="canonical_key",
        how="left",
    )
    examples.to_csv(RESULTS / "primary_reference_prototype_examples.csv", index=False)

    path = RESULTS / "results_for_manuscript.json"
    results = json.loads(path.read_text(encoding="utf-8"))
    results["primary_candidate_analysis"] = primary
    results["candidate_stability"] = stability
    results["positive_control_dual_g3"] = controls
    results["extended_validation"] = extended
    results["candidate_uncertainty"] = uncertainty
    results["screening"]["primary_radius_policy"] = {"name": "positive-control P95", "threshold": p95}
    results["screening"]["nested_conservative_radius_policy"] = {"name": "positive-control P90", "threshold": p90}
    results["screening"].pop("companion_radius_policy", None)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return primary


def main() -> None:
    arm = reference_arm()
    p90, p95, known = thresholds()
    reference = screen_reference(arm, p90, p95)
    stability = stability_summary(arm, p90, p95)
    controls = positive_control_detail(p90, p95, known)
    make_li_tm_window_sensitivity()
    make_literature_context_table()
    primary = update_results(reference, stability, controls, arm, p90, p95)
    print(json.dumps(primary, indent=2))


if __name__ == "__main__":
    main()
