#!/usr/bin/env python3
"""Create paired inference, primary shortlist tiers, and manuscript-ready tables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from rebuild_data_and_models import model_factories
except ModuleNotFoundError:  # Lightweight document/figure-only rebuild environment.
    model_factories = None


ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "paper_release_v1"
DATA_DIR = RELEASE / "data"
RESULTS_DIR = RELEASE / "results"
SEARCH_DIR = RESULTS_DIR / "search_candidates"
CONFIG_DIR = RELEASE / "config"
SEED = 42
BOOTSTRAP_REPS = 10_000
PERMUTATION_REPS = 100_000

PROTO_FAMILY = {
    "Li-rich-NMC": "layered",
    "NMC-532": "layered",
    "NMC-622": "layered",
    "NMC-811": "layered",
    "NCA": "layered",
    "LiNiO2": "layered",
    "LiCoO2": "layered",
    "Li2MnO3": "layered_lirich",
    "LiMn2O4": "spinel",
    "LNMO-spinel": "spinel",
    "LiFePO4": "olivine",
    "LiMnPO4": "olivine",
}

LI_TM_WINDOW = {
    "layered": (0.80, 1.50),
    "layered_lirich": (1.50, 2.20),
    "spinel": (0.40, 0.60),
    "olivine": (0.80, 1.20),
}


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "spearman": float(spearmanr(y_true, y_pred).statistic),
    }


def clustered_sign_flip_p(
    y_true: np.ndarray,
    left_pred: np.ndarray,
    right_pred: np.ndarray,
    groups: np.ndarray,
    repetitions: int = PERMUTATION_REPS,
    seed: int = 20260719,
) -> float:
    """Two-sided paired permutation p value with canonical composition as unit.

    For a common target vector, the R2 difference is proportional to the
    right-minus-left squared-error difference.  Swapping the two predictions
    for every row in a canonical composition is therefore equivalent to a
    sign flip of that composition's summed squared-error contrast.
    """

    unique_groups = np.unique(groups)
    contributions = np.asarray(
        [
            np.sum(
                (y_true[groups == group] - right_pred[groups == group]) ** 2
                - (y_true[groups == group] - left_pred[groups == group]) ** 2
            )
            for group in unique_groups
        ],
        dtype=float,
    )
    observed = float(contributions.sum())
    rng = np.random.default_rng(seed)
    exceedances = 0
    batch_size = 5_000
    for start in range(0, repetitions, batch_size):
        size = min(batch_size, repetitions - start)
        signs = rng.choice((-1.0, 1.0), size=(size, len(contributions)))
        permuted = signs @ contributions
        exceedances += int(np.count_nonzero(np.abs(permuted) >= abs(observed) - 1e-12))
    return float((exceedances + 1) / (repetitions + 1))


def holm_adjust(raw_p_values: dict[str, float]) -> dict[str, float]:
    """Holm family-wise-error adjustment with monotonicity enforcement."""

    ordered = sorted(raw_p_values, key=raw_p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, name in enumerate(ordered):
        candidate = min(1.0, (total - rank) * float(raw_p_values[name]))
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def make_evaluation_protocol_gap() -> dict[str, object]:
    """Quantify record-split optimism relative to exact canonical holdout."""

    record = pd.read_csv(RESULTS_DIR / "oof_record_cv_catboost_context.csv").sort_values("row_id")
    heldout_all = pd.read_csv(RESULTS_DIR / "oof_canonical_composition_heldout_all_models.csv")
    heldout = heldout_all[
        heldout_all["model"].eq("catboost__composition_plus_context")
    ].sort_values("row_id")
    if not np.array_equal(record["row_id"].to_numpy(), heldout["row_id"].to_numpy()):
        raise AssertionError("Record-CV and canonical-holdout rows are not aligned")
    if not np.allclose(record["y_true"], heldout["y_true"]):
        raise AssertionError("Record-CV and canonical-holdout targets differ")

    y_true = record["y_true"].to_numpy(float)
    record_pred = record["y_pred"].to_numpy(float)
    heldout_pred = heldout["y_pred"].to_numpy(float)
    groups = record["canonical_key"].to_numpy(str)
    unique_groups = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique_groups}
    point = float(r2_score(y_true, record_pred) - r2_score(y_true, heldout_pred))
    rng = np.random.default_rng(SEED)
    bootstrap = np.empty(BOOTSTRAP_REPS, dtype=float)
    for rep in range(BOOTSTRAP_REPS):
        sampled_groups = rng.choice(unique_groups, len(unique_groups), replace=True)
        indices = np.concatenate([by_group[group] for group in sampled_groups])
        bootstrap[rep] = r2_score(y_true[indices], record_pred[indices]) - r2_score(
            y_true[indices], heldout_pred[indices]
        )
    p_value = clustered_sign_flip_p(y_true, record_pred, heldout_pred, groups)
    output = {
        "comparison": "record_level_cv_minus_exact_canonical_composition_holdout",
        "record_level_r2": float(r2_score(y_true, record_pred)),
        "canonical_composition_heldout_r2": float(r2_score(y_true, heldout_pred)),
        "delta_r2": point,
        "cluster_bootstrap_ci95": [
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ],
        "bootstrap_unit": "canonical_composition",
        "bootstrap_repetitions": BOOTSTRAP_REPS,
        "clustered_sign_flip_p_two_sided": p_value,
        "permutation_unit": "canonical_composition",
        "permutation_repetitions": PERMUTATION_REPS,
    }
    (RESULTS_DIR / "evaluation_protocol_gap.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    pd.DataFrame([output]).to_csv(RESULTS_DIR / "evaluation_protocol_gap.csv", index=False)
    return output


def make_inference_family(
    pairwise: pd.DataFrame, evaluation_gap: dict[str, object]
) -> list[dict[str, object]]:
    """Define and adjust the four primary formal tests used in the revision.

    The family is explicitly a post-review specification.  Threshold sweeps,
    secondary metrics, and the five-seed search summary remain descriptive and
    are not silently treated as additional hypothesis tests.
    """

    oof = pd.read_csv(RESULTS_DIR / "oof_canonical_composition_heldout_all_models.csv")
    truth = (
        oof[["row_id", "canonical_key", "y_true"]]
        .drop_duplicates("row_id")
        .sort_values("row_id")
    )
    predictions = oof.pivot(index="row_id", columns="model", values="y_pred").sort_index()
    y_true = truth["y_true"].to_numpy(float)
    groups = truth["canonical_key"].to_numpy(str)

    model_p = clustered_sign_flip_p(
        y_true,
        predictions["catboost__composition_plus_context"].to_numpy(float),
        predictions["random_forest__composition_plus_context"].to_numpy(float),
        groups,
    )
    context_p = clustered_sign_flip_p(
        y_true,
        predictions["catboost__composition_plus_context"].to_numpy(float),
        predictions["catboost__composition_only"].to_numpy(float),
        groups,
    )
    distance = json.loads(
        (RESULTS_DIR / "distance_error_inference.json").read_text(encoding="utf-8")
    )
    raw = {
        "evaluation_unit_optimism_gap": float(
            evaluation_gap["clustered_sign_flip_p_two_sided"]
        ),
        "measurement_context_ablation": context_p,
        "catboost_vs_random_forest": model_p,
        "distance_error_association": float(
            distance["formula_label_permutation_p_two_sided"]
        ),
    }
    adjusted = holm_adjust(raw)

    rho = float(distance["spearman_rho"])
    n = int(distance["n_canonical_compositions"])
    z_alpha = float(norm.ppf(0.975))
    noncentrality = float(np.arctanh(abs(rho)) * np.sqrt(n - 3))
    approximate_power = float(
        norm.cdf(-z_alpha - noncentrality) + 1 - norm.cdf(z_alpha - noncentrality)
    )
    minimum_detectable_rho_80 = float(
        np.tanh((z_alpha + norm.ppf(0.80)) / np.sqrt(n - 3))
    )

    effect_lookup = {
        "evaluation_unit_optimism_gap": float(evaluation_gap["delta_r2"]),
        "measurement_context_ablation": float(
            pairwise.loc[
                (pairwise["comparison"] == "catboost_context_vs_catboost_composition_only")
                & (pairwise["metric"] == "r2"),
                "paired_delta_left_minus_right",
            ].iloc[0]
        ),
        "catboost_vs_random_forest": float(
            pairwise.loc[
                (pairwise["comparison"] == "catboost_context_vs_random_forest_context")
                & (pairwise["metric"] == "r2"),
                "paired_delta_left_minus_right",
            ].iloc[0]
        ),
        "distance_error_association": rho,
    }
    effect_metric = {
        "evaluation_unit_optimism_gap": "delta_r2",
        "measurement_context_ablation": "delta_r2",
        "catboost_vs_random_forest": "delta_r2",
        "distance_error_association": "spearman_rho",
    }
    rows = [
        {
            "family": "post_review_primary_inference_family",
            "test": name,
            "effect_metric": effect_metric[name],
            "effect_estimate": effect_lookup[name],
            "raw_p_two_sided": raw[name],
            "holm_adjusted_p": adjusted[name],
            "familywise_alpha": 0.05,
            "inference_unit": "canonical_composition",
        }
        for name in raw
    ]
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "primary_inference_family.csv", index=False)
    power = {
        "distance_error_observed_rho": rho,
        "n_canonical_compositions": n,
        "two_sided_alpha": 0.05,
        "fisher_z_approximate_power_at_observed_rho": approximate_power,
        "fisher_z_approximate_minimum_detectable_abs_rho_for_80_percent_power": minimum_detectable_rho_80,
        "interpretation": "The observed weak positive association is statistically inconclusive; the analysis is not powered to establish absence of weak association.",
    }
    (RESULTS_DIR / "distance_error_power.json").write_text(
        json.dumps(power, indent=2), encoding="utf-8"
    )
    return rows


def paired_cluster_bootstrap(oof: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        (
            "catboost_context_vs_random_forest_context",
            "catboost__composition_plus_context",
            "random_forest__composition_plus_context",
        ),
        (
            "catboost_context_vs_catboost_composition_only",
            "catboost__composition_plus_context",
            "catboost__composition_only",
        ),
        (
            "random_forest_context_vs_ridge_context",
            "random_forest__composition_plus_context",
            "ridge__composition_plus_context",
        ),
    ]
    truth = (
        oof[["row_id", "canonical_key", "y_true"]]
        .drop_duplicates("row_id")
        .sort_values("row_id")
        .reset_index(drop=True)
    )
    predictions = oof.pivot(index="row_id", columns="model", values="y_pred").sort_index()
    if not np.array_equal(predictions.index.to_numpy(), truth["row_id"].to_numpy()):
        raise AssertionError("OOF row alignment failed")

    y_true = truth["y_true"].to_numpy(float)
    groups = truth["canonical_key"].to_numpy(str)
    unique_groups = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(SEED)
    sampled_indices = [
        np.concatenate([by_group[group] for group in rng.choice(unique_groups, len(unique_groups), replace=True)])
        for _ in range(BOOTSTRAP_REPS)
    ]

    rows: list[dict[str, object]] = []
    for comparison, left, right in comparisons:
        left_pred = predictions[left].to_numpy(float)
        right_pred = predictions[right].to_numpy(float)
        left_point = metrics(y_true, left_pred)
        right_point = metrics(y_true, right_pred)
        boot = {name: np.empty(BOOTSTRAP_REPS, dtype=float) for name in left_point}
        for rep, indices in enumerate(sampled_indices):
            left_metrics = metrics(y_true[indices], left_pred[indices])
            right_metrics = metrics(y_true[indices], right_pred[indices])
            for name in boot:
                boot[name][rep] = left_metrics[name] - right_metrics[name]
        for name in left_point:
            values = boot[name][np.isfinite(boot[name])]
            rows.append(
                {
                    "comparison": comparison,
                    "left_model": left,
                    "right_model": right,
                    "metric": name,
                    "left_value": left_point[name],
                    "right_value": right_point[name],
                    "paired_delta_left_minus_right": left_point[name] - right_point[name],
                    "paired_delta_ci95_low": float(np.percentile(values, 2.5)),
                    "paired_delta_ci95_high": float(np.percentile(values, 97.5)),
                    "bootstrap_unit": "canonical_composition",
                    "bootstrap_repetitions": BOOTSTRAP_REPS,
                }
            )
    return pd.DataFrame(rows)


def make_alias_and_integrity_tables() -> None:
    groups = pd.read_csv(DATA_DIR / "formula_group_summary.csv")
    aliases = groups[groups["n_raw_strings"] > 1].copy()
    aliases.to_csv(DATA_DIR / "formula_alias_groups.csv", index=False)

    audit = pd.read_csv(DATA_DIR / "formula_canonicalization_audit.csv")
    integrity = audit[
        audit[
            [
                "raw_vs_extracted_mismatch",
                "raw_vs_legacy_feature_mismatch",
                "legacy_zero_element_vector",
                "carbon_in_extracted_name",
            ]
        ].any(axis=1)
    ].copy()
    integrity.to_csv(DATA_DIR / "legacy_identity_feature_integrity_flags.csv", index=False)


def export_baseline_parameters() -> None:
    if model_factories is None:
        expected = [
            CONFIG_DIR / "dummy_median_params.json",
            CONFIG_DIR / "ridge_params.json",
            CONFIG_DIR / "random_forest_params.json",
            CONFIG_DIR / "catboost_params.json",
        ]
        missing = [path.name for path in expected if not path.exists()]
        if missing:
            raise RuntimeError(
                "Model dependencies are unavailable and fitted parameter exports are missing: "
                + ", ".join(missing)
            )
        return

    def serializable(value: object) -> object:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple)):
            return [serializable(item) for item in value]
        return str(value)

    for name, factory in model_factories().items():
        estimator = factory()
        parameters = {key: serializable(value) for key, value in estimator.get_params(deep=True).items()}
        (CONFIG_DIR / f"{name}_params.json").write_text(
            json.dumps(parameters, indent=2), encoding="utf-8"
        )


def make_primary_shortlist() -> dict[str, object]:
    decision = json.loads((RESULTS_DIR / "nsga_decision.json").read_text(encoding="utf-8"))
    primary_arm = (
        "nsga2__floor_on"
        if decision["retain_nsga2_as_substantive_performance_claim"]
        else "random__floor_on"
    )
    primary_reason = (
        "Matched five-seed feasible-hypervolume result met the revision reporting rule for an NSGA-II benefit claim."
        if primary_arm.startswith("nsga2")
        else "NSGA-II did not meet the revision's conservative five-seed reporting rule; the simpler random-search plus prototype-floor arm is primary."
    )
    source = SEARCH_DIR / f"{primary_arm}_seed{SEED}_topk_screened.csv"
    screened = pd.read_csv(source)
    if len(screened) != 500:
        raise AssertionError("Primary reference TopK must contain 500 unique compositions")
    if "parameter_bounds_ok" in screened and not screened["parameter_bounds_ok"].astype(bool).all():
        raise AssertionError("Primary reference TopK contains an out-of-bounds candidate")

    known_summary = json.loads(
        (RESULTS_DIR / "known_control_summary.json").read_text(encoding="utf-8")
    )
    control_p95 = float(known_summary["positive_control_radius_p95_after_g1_g2"])
    pass_g1 = screened["g1_pass"].astype(bool)
    pass_g2 = screened["g2_pass"].astype(bool)
    pass_g4 = screened["g4_pass"].astype(bool)
    radius = pd.to_numeric(screened["radius_mismatch"], errors="coerce")
    screened["pass_control_anchored_g3"] = radius.le(control_p95)
    screened["pass_g1_g2_g3_control_anchored"] = (
        pass_g1 & pass_g2 & screened["pass_control_anchored_g3"]
    )
    screened["pass_g1_g2_g3_g4_control_anchored"] = (
        screened["pass_g1_g2_g3_control_anchored"] & pass_g4
    )
    screened["dual_operating_tier"] = np.select(
        [
            pass_g1 & pass_g2 & pass_g4 & radius.le(0.15),
            pass_g1 & pass_g2 & pass_g4 & radius.gt(0.15) & radius.le(control_p95),
            pass_g1 & pass_g2 & pass_g4 & radius.gt(control_p95),
            pass_g1 & pass_g2 & ~pass_g4 & radius.le(control_p95),
            pass_g1 & pass_g2 & ~pass_g4 & radius.gt(control_p95),
            pass_g1 & ~pass_g2,
            ~pass_g1,
        ],
        [
            "A_high_conservatism_core_le_0.15",
            "B_control_anchored_extension_0.15_to_P95",
            "C_beyond_control_P95_radius_risk",
            "C2_redox_risk_within_control_P95",
            "C3_radius_and_redox_risk_beyond_control_P95",
            "D_Li_TM_window_risk",
            "E_formal_ionicity_risk",
        ],
        default="Unclassified",
    )
    # Compatibility alias for consumers of the previous release.  The new
    # dual_operating_tier is the authoritative interpretation.
    screened["risk_tier"] = screened["dual_operating_tier"]
    screened.to_csv(RESULTS_DIR / "primary_reference_topk_risk_tiers.csv", index=False)
    core = screened[pass_g1 & pass_g2 & pass_g4 & radius.le(0.15)].copy()
    core = core.sort_values(
        ["proto_name", "y_discounted", "canonical_key"], ascending=[True, False, True]
    )
    portfolio = screened[screened["pass_g1_g2_g3_g4_control_anchored"]].copy()
    portfolio = portfolio.sort_values(
        ["proto_name", "y_discounted", "canonical_key"], ascending=[True, False, True]
    )
    core.to_csv(RESULTS_DIR / "primary_reference_strict_shortlist.csv", index=False)
    core.to_csv(RESULTS_DIR / "primary_reference_high_conservatism_core.csv", index=False)
    portfolio.to_csv(RESULTS_DIR / "primary_reference_control_anchored_portfolio.csv", index=False)
    examples = core.groupby("proto_name", as_index=False, group_keys=False).head(1)
    examples.to_csv(RESULTS_DIR / "primary_reference_prototype_examples.csv", index=False)

    summary = pd.read_csv(RESULTS_DIR / "search_seed_summary.csv")
    primary_seeds = summary[summary["arm"] == primary_arm]
    output = {
        "primary_arm": primary_arm,
        "reference_seed": SEED,
        "selection_rule": "seed 42 is frozen as the exact reference release; cross-seed candidate membership is reported separately as a robustness analysis",
        "reason": primary_reason,
        "reference_topk": int(len(screened)),
        "positive_control_anchored_radius_p95": control_p95,
        "reference_control_anchored_survivors": int(len(portfolio)),
        "reference_strict_0_15_survivors": int(len(core)),
        "reference_control_anchored_extension": int(len(portfolio) - len(core)),
        "reference_prototype_coverage": int(screened["proto_name"].nunique()),
        "control_anchored_prototype_coverage": int(portfolio["proto_name"].nunique()),
        "high_conservatism_core_prototype_coverage": int(core["proto_name"].nunique()),
        "risk_tier_counts": {str(key): int(value) for key, value in screened["dual_operating_tier"].value_counts().items()},
        "five_seed_strict_survivors_mean": float(primary_seeds["g4_survivors"].mean()),
        "five_seed_strict_survivors_sd": float(primary_seeds["g4_survivors"].std(ddof=1)),
        "five_seed_raw_prediction_mean": float(primary_seeds["y_pred_mean"].mean()),
        "five_seed_raw_prediction_sd": float(primary_seeds["y_pred_mean"].std(ddof=1)),
        "five_seed_discounted_prediction_mean": float(primary_seeds["y_discounted_mean"].mean()),
        "five_seed_discounted_prediction_sd": float(primary_seeds["y_discounted_mean"].std(ddof=1)),
    }
    (RESULTS_DIR / "primary_analysis_summary.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    return output


def make_candidate_stability(primary: dict[str, object]) -> dict[str, object]:
    """Report candidate membership stability within the five-seed reference arm."""

    arm = str(primary["primary_arm"])
    p95 = float(primary["positive_control_anchored_radius_p95"])
    frames: list[pd.DataFrame] = []
    per_seed: list[dict[str, int]] = []
    for seed in range(42, 47):
        frame = pd.read_csv(SEARCH_DIR / f"{arm}_seed{seed}_topk_screened.csv").copy()
        frame["seed"] = seed
        pass_g1_g2_g4 = (
            frame["g1_pass"].astype(bool)
            & frame["g2_pass"].astype(bool)
            & frame["g4_pass"].astype(bool)
        )
        radius = pd.to_numeric(frame["radius_mismatch"], errors="coerce")
        frame["high_conservatism_core"] = pass_g1_g2_g4 & radius.le(0.15)
        frame["control_anchored_portfolio"] = pass_g1_g2_g4 & radius.le(p95)
        per_seed.append(
            {
                "seed": seed,
                "topk": int(len(frame)),
                "high_conservatism_core": int(frame["high_conservatism_core"].sum()),
                "control_anchored_portfolio": int(frame["control_anchored_portfolio"].sum()),
            }
        )
        frames.append(frame)
    all_rows = pd.concat(frames, ignore_index=True)

    records: list[dict[str, object]] = []
    for canonical_key, part in all_rows.groupby("canonical_key", sort=True):
        first = part.sort_values("seed").iloc[0]
        topk_seeds = sorted(part["seed"].astype(int).unique().tolist())
        control_seeds = sorted(
            part.loc[part["control_anchored_portfolio"], "seed"].astype(int).unique().tolist()
        )
        core_seeds = sorted(
            part.loc[part["high_conservatism_core"], "seed"].astype(int).unique().tolist()
        )
        records.append(
            {
                "canonical_key": canonical_key,
                "Formula": first["Formula"],
                "proto_name": first["proto_name"],
                "topk_seed_count": len(topk_seeds),
                "control_anchored_seed_count": len(control_seeds),
                "high_conservatism_seed_count": len(core_seeds),
                "topk_seeds": ";".join(map(str, topk_seeds)),
                "control_anchored_seeds": ";".join(map(str, control_seeds)),
                "high_conservatism_seeds": ";".join(map(str, core_seeds)),
                "control_anchored_majority_consensus": len(control_seeds) >= 3,
                "high_conservatism_majority_consensus": len(core_seeds) >= 3,
            }
        )
    stability = pd.DataFrame(records)
    stability.to_csv(RESULTS_DIR / "primary_candidate_stability.csv", index=False)

    def jaccard(sets: list[set[str]]) -> float:
        values = []
        for left in range(len(sets)):
            for right in range(left + 1, len(sets)):
                union = sets[left] | sets[right]
                values.append(len(sets[left] & sets[right]) / len(union) if union else np.nan)
        return float(np.nanmean(values))

    topk_sets = [set(frame["canonical_key"]) for frame in frames]
    control_sets = [
        set(frame.loc[frame["control_anchored_portfolio"], "canonical_key"]) for frame in frames
    ]
    core_sets = [
        set(frame.loc[frame["high_conservatism_core"], "canonical_key"]) for frame in frames
    ]
    output = {
        "primary_arm": arm,
        "seeds": [42, 43, 44, 45, 46],
        "consensus_definition": "canonical candidate appears in at least three of five reference-arm seeds",
        "per_seed_counts": per_seed,
        "topk_union": int(stability["topk_seed_count"].gt(0).sum()),
        "topk_all_five": int(stability["topk_seed_count"].eq(5).sum()),
        "topk_majority_consensus": int(stability["topk_seed_count"].ge(3).sum()),
        "topk_mean_pairwise_jaccard": jaccard(topk_sets),
        "control_anchored_union": int(stability["control_anchored_seed_count"].gt(0).sum()),
        "control_anchored_all_five": int(stability["control_anchored_seed_count"].eq(5).sum()),
        "control_anchored_majority_consensus": int(stability["control_anchored_seed_count"].ge(3).sum()),
        "control_anchored_mean_pairwise_jaccard": jaccard(control_sets),
        "high_conservatism_union": int(stability["high_conservatism_seed_count"].gt(0).sum()),
        "high_conservatism_all_five": int(stability["high_conservatism_seed_count"].eq(5).sum()),
        "high_conservatism_majority_consensus": int(stability["high_conservatism_seed_count"].ge(3).sum()),
        "high_conservatism_mean_pairwise_jaccard": jaccard(core_sets),
    }
    (RESULTS_DIR / "candidate_stability_summary.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    return output


def make_positive_control_detail(primary: dict[str, object]) -> dict[str, object]:
    """Export the complete dual G3 curve and prototype-resolved denominators."""

    known = pd.read_csv(RESULTS_DIR / "known_control_screened.csv").copy()
    p95 = float(primary["positive_control_anchored_radius_p95"])
    reached = known["g1_pass"].astype(bool) & known["g2_pass"].astype(bool)
    radius = pd.to_numeric(known["radius_mismatch"], errors="coerce")
    g4 = known["g4_pass"].astype(bool)
    known["g3_high_conservatism_pass"] = radius.le(0.15)
    known["g3_control_anchored_pass"] = radius.le(p95)
    known["first_failure_control_anchored"] = np.select(
        [
            ~known["g1_pass"].astype(bool),
            known["g1_pass"].astype(bool) & ~known["g2_pass"].astype(bool),
            reached & ~known["g3_control_anchored_pass"],
            reached & known["g3_control_anchored_pass"] & ~g4,
        ],
        ["G1", "G2", "G3", "G4"],
        default="Pass",
    )
    known.to_csv(RESULTS_DIR / "known_control_screened.csv", index=False)

    rows: list[dict[str, object]] = []
    for proto, part in known.groupby("proto_name", sort=True):
        part_reached = part[part["g1_pass"].astype(bool) & part["g2_pass"].astype(bool)]
        rows.append(
            {
                "proto_name": proto,
                "applicable_n": int(len(part)),
                "g1_survivors_n": int(part["g1_pass"].sum()),
                "g1_g2_survivors_n": int(part["pass_g1_g2"].sum()),
                "g3_high_conservatism_0_15_survivors_n": int(
                    (part["pass_g1_g2"].astype(bool) & part["g3_high_conservatism_pass"]).sum()
                ),
                "g3_high_conservatism_0_15_losses_after_g1_g2_n": int(
                    (~part_reached["g3_high_conservatism_pass"]).sum()
                ),
                "g3_control_anchored_p95_survivors_n": int(
                    (part["pass_g1_g2"].astype(bool) & part["g3_control_anchored_pass"]).sum()
                ),
                "g3_control_anchored_p95_losses_after_g1_g2_n": int(
                    (~part_reached["g3_control_anchored_pass"]).sum()
                ),
                "final_high_conservatism_survivors_n": int(
                    (part["pass_g1_g2"].astype(bool) & part["g3_high_conservatism_pass"] & part["g4_pass"].astype(bool)).sum()
                ),
                "final_control_anchored_survivors_n": int(
                    (part["pass_g1_g2"].astype(bool) & part["g3_control_anchored_pass"] & part["g4_pass"].astype(bool)).sum()
                ),
            }
        )
    prototype = pd.DataFrame(rows).sort_values(
        ["applicable_n", "proto_name"], ascending=[False, True]
    )
    prototype.to_csv(RESULTS_DIR / "positive_control_prototype_summary.csv", index=False)

    summary = {
        "positive_control_radius_p95_after_g1_g2": p95,
        "g1_g2_denominator": int(reached.sum()),
        "high_conservatism_0_15_g3_survivors": int((reached & radius.le(0.15)).sum()),
        "high_conservatism_0_15_conditional_retention": float(
            (reached & radius.le(0.15)).sum() / reached.sum()
        ),
        "high_conservatism_0_15_final_survivors": int(
            (reached & radius.le(0.15) & g4).sum()
        ),
        "control_anchored_p95_g3_survivors": int((reached & radius.le(p95)).sum()),
        "control_anchored_p95_conditional_retention": float(
            (reached & radius.le(p95)).sum() / reached.sum()
        ),
        "control_anchored_p95_final_survivors": int(
            (reached & radius.le(p95) & g4).sum()
        ),
    }
    (RESULTS_DIR / "positive_control_dual_g3_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def make_literature_context_table() -> None:
    """Machine-readable, explicitly non-leaderboard context for reported metrics."""

    rows = [
        {
            "study": "Min et al. 2018",
            "dataset_scope": "330 controlled Ni-rich NCM experimental datasets",
            "inputs": "13 synthesis, ICP-MS, and XRD variables",
            "evaluation": "random 80/20 split repeated over 300 training selections",
            "target": "initial discharge capacity",
            "reported_r2": 0.751,
            "reported_mae_mAh_g": 2.84,
            "reported_rmse_mAh_g": np.nan,
            "comparability_note": "Narrow controlled chemistry and richer measured inputs; experimentally validated optimization task; not directly comparable.",
            "doi": "10.1038/s41598-018-34201-4",
        },
        {
            "study": "Wang et al. 2021",
            "dataset_scope": "168 distinct doped NCM systems",
            "inputs": "20 composition, structure, dopant, and protocol covariates",
            "evaluation": "random 4:1 train/test split; fivefold tuning within training",
            "target": "initial discharge capacity",
            "reported_r2": 0.76,
            "reported_mae_mAh_g": np.nan,
            "reported_rmse_mAh_g": 16.66,
            "comparability_note": "Narrow NCM task with lattice and voltage-window inputs; not directly comparable.",
            "doi": "10.1021/acscentsci.1c00611",
        },
        {
            "study": "This work, secondary record CV",
            "dataset_scope": "907 heterogeneous literature records; 171 canonical compositions",
            "inputs": "composition descriptors plus four available context variables",
            "evaluation": "stratified tenfold record-level CV",
            "target": "literature-reported discharge capacity",
            "reported_r2": 0.7049890878835565,
            "reported_mae_mAh_g": 20.766980218861738,
            "reported_rmse_mAh_g": 28.45674874595885,
            "comparability_note": "Secondary interpolation diagnostic only.",
            "doi": "",
        },
        {
            "study": "This work, primary exact canonical holdout",
            "dataset_scope": "907 heterogeneous literature records; 171 canonical compositions",
            "inputs": "composition descriptors plus four available context variables",
            "evaluation": "exact leave-one-canonical-composition-out",
            "target": "literature-reported discharge capacity",
            "reported_r2": 0.5271508689429464,
            "reported_mae_mAh_g": 26.438779104640753,
            "reported_rmse_mAh_g": 36.026948102674826,
            "comparability_note": "Primary unseen-composition estimate; demonstrates evaluation-unit sensitivity rather than leaderboard superiority.",
            "doi": "",
        },
    ]
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "literature_metric_context.csv", index=False)


def make_li_tm_window_sensitivity() -> None:
    """Export the original Figure-5-style G2 window sweep from frozen TopK files.

    The sweep is conditional on reaching G2 (passing G1).  It is descriptive:
    the implemented width scale remains 1.0, and no width is calibrated as a
    universal chemical boundary.
    """

    widths = np.round(np.arange(0.5, 1.501, 0.1), 2)
    rows: list[dict[str, object]] = []

    def append_pool(frame: pd.DataFrame, pool: str, seed: int | str) -> None:
        reached = frame["g1_pass"].astype(bool)
        ratios = pd.to_numeric(frame["li_tm_ratio"], errors="coerce")
        families = frame["proto_name"].map(PROTO_FAMILY)
        for width in widths:
            passed = np.zeros(len(frame), dtype=bool)
            for family, (low, high) in LI_TM_WINDOW.items():
                center = (low + high) / 2.0
                half = (high - low) / 2.0 * float(width)
                mask = families.eq(family) & ratios.between(center - half, center + half, inclusive="both")
                passed |= mask.to_numpy(bool)
            n_reaching = int(reached.sum())
            n_passing = int((reached.to_numpy(bool) & passed).sum())
            rows.append(
                {
                    "pool": pool,
                    "seed": seed,
                    "relative_li_tm_window_width": float(width),
                    "n_reaching_g2": n_reaching,
                    "n_passing_g2": n_passing,
                    "conditional_retention": float(n_passing / n_reaching) if n_reaching else np.nan,
                }
            )

    summary = pd.read_csv(RESULTS_DIR / "search_seed_summary.csv")
    for item in summary[["arm", "seed"]].drop_duplicates().itertuples(index=False):
        path = SEARCH_DIR / f"{item.arm}_seed{int(item.seed)}_topk_screened.csv"
        append_pool(pd.read_csv(path), str(item.arm), int(item.seed))

    append_pool(
        pd.read_csv(RESULTS_DIR / "known_control_screened.csv"),
        "known_positive_controls",
        "control",
    )
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "li_tm_window_sensitivity.csv", index=False)


def update_manuscript_results(
    pairwise: pd.DataFrame,
    primary: dict[str, object],
    evaluation_gap: dict[str, object],
    inference_family: list[dict[str, object]],
    stability: dict[str, object],
    positive_control_dual_g3: dict[str, object],
) -> None:
    path = RESULTS_DIR / "results_for_manuscript.json"
    results = json.loads(path.read_text(encoding="utf-8"))
    results["paired_model_comparisons"] = pairwise.to_dict(orient="records")
    results["primary_candidate_analysis"] = primary
    results["evaluation_protocol_gap"] = evaluation_gap
    results["primary_inference_family"] = inference_family
    results["candidate_stability"] = stability
    results["positive_control_dual_g3"] = positive_control_dual_g3
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def main() -> None:
    make_alias_and_integrity_tables()
    export_baseline_parameters()
    oof = pd.read_csv(RESULTS_DIR / "oof_canonical_composition_heldout_all_models.csv")
    pairwise = paired_cluster_bootstrap(oof)
    pairwise.to_csv(RESULTS_DIR / "paired_model_cluster_bootstrap.csv", index=False)
    evaluation_gap = make_evaluation_protocol_gap()
    inference_family = make_inference_family(pairwise, evaluation_gap)
    primary = make_primary_shortlist()
    stability = make_candidate_stability(primary)
    positive_control_dual_g3 = make_positive_control_detail(primary)
    make_literature_context_table()
    make_li_tm_window_sensitivity()
    update_manuscript_results(
        pairwise,
        primary,
        evaluation_gap,
        inference_family,
        stability,
        positive_control_dual_g3,
    )
    print(json.dumps(primary, indent=2))


if __name__ == "__main__":
    main()
