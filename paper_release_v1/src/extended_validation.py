#!/usr/bin/env python3
"""Deployment-aligned validation and candidate-stability analysis.

This module deliberately reuses the completed canonical OOF predictions and the
matched 20-seed search outputs.  It adds only analyses that were absent from the
earlier release: an exact nominal-context slice, leave-one-structural-family-out
testing, candidate recurrence, model-resampling uncertainty, context robustness,
and element-support diagnostics.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from pymatgen.core import Composition
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from rebuild_data_and_models import CB_PARAMS
from search_screen_benchmark import CandidateEvaluator, PERIODIC_ELEMENTS


ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "paper_release_v1"
DATA = RELEASE / "data"
RESULTS = RELEASE / "results"
SEARCH = RESULTS / "search_candidates"
MODELS = RELEASE / "models"
BOOTSTRAP_REPS = 20_000
MODEL_BOOTSTRAP_REPS = 100
SEEDS = list(range(42, 62))

STRUCTURAL_FAMILY = {
    "Li-rich-NMC": "layered",
    "NMC-532": "layered",
    "NMC-622": "layered",
    "NMC-811": "layered",
    "NCA": "layered",
    "LiNiO2": "layered",
    "LiCoO2": "layered",
    "Li2MnO3": "layered",
    "LiMn2O4": "spinel",
    "LNMO-spinel": "spinel",
    "LiFePO4": "olivine",
    "LiMnPO4": "olivine",
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "spearman": float(spearmanr(y_true, y_pred).statistic),
    }


def clustered_metric_ci(
    frame: pd.DataFrame,
    *,
    repetitions: int = BOOTSTRAP_REPS,
    seed: int = 42011,
) -> dict[str, list[float]]:
    groups = frame["canonical_key"].astype(str).unique()
    by_group = {
        group: np.flatnonzero(frame["canonical_key"].astype(str).to_numpy() == group)
        for group in groups
    }
    y_true = frame["y_true"].to_numpy(float)
    y_pred = frame["y_pred"].to_numpy(float)
    rng = np.random.default_rng(seed)
    values = {name: np.empty(repetitions, dtype=float) for name in ["r2", "mae", "rmse", "spearman"]}
    for repetition in range(repetitions):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([by_group[group] for group in sampled])
        metrics = point_metrics(y_true[indices], y_pred[indices])
        for name, value in metrics.items():
            values[name][repetition] = value
    return {
        name: [
            float(np.nanpercentile(array, 2.5)),
            float(np.nanpercentile(array, 97.5)),
        ]
        for name, array in values.items()
    }


def nominal_context_validation() -> dict[str, object]:
    raw = pd.read_csv(DATA / "clean_rows.csv").sort_values("row_id").reset_index(drop=True)
    oof = pd.read_csv(RESULTS / "oof_canonical_composition_heldout_all_models.csv")
    oof = oof[oof["model"].eq("catboost__composition_plus_context")].sort_values("row_id").reset_index(drop=True)
    if not np.array_equal(raw["row_id"].to_numpy(), oof["row_id"].to_numpy()):
        raise AssertionError("Nominal-context OOF rows are not aligned")
    mask = (
        np.isclose(pd.to_numeric(raw["temperature"]), 298.15)
        & np.isclose(pd.to_numeric(raw["current_rate"]), 0.5)
        & np.isclose(pd.to_numeric(raw["cycle_value"]), 1.0)
    )
    nominal = oof.loc[mask, ["row_id", "canonical_key", "y_true", "y_pred"]].copy()
    nominal.insert(2, "temperature_K", 298.15)
    nominal.insert(3, "current_rate_C", 0.5)
    nominal.insert(4, "cycle_number", 1.0)
    nominal.to_csv(RESULTS / "nominal_context_slice_predictions.csv", index=False)
    metrics = point_metrics(nominal["y_true"].to_numpy(float), nominal["y_pred"].to_numpy(float))
    ci = clustered_metric_ci(nominal)
    output = {
        "definition": "exact observed records at 298.15 K, 0.5 C, cycle 1 evaluated with canonical-composition-held-out OOF predictions",
        "n_records": int(len(nominal)),
        "n_canonical_compositions": int(nominal["canonical_key"].nunique()),
        **metrics,
        "cluster_bootstrap_ci95": ci,
        "bootstrap_unit": "canonical_composition",
        "bootstrap_repetitions": BOOTSTRAP_REPS,
    }
    (RESULTS / "nominal_context_validation.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def leave_one_family_out_validation() -> dict[str, object]:
    raw = pd.read_csv(DATA / "clean_rows.csv").sort_values("row_id").reset_index(drop=True)
    features = pd.read_csv(DATA / "features_composition_context.csv")
    assignments = pd.read_csv(RESULTS / "known_control_prototype_assignment.csv")
    assignment = assignments.dropna(subset=["proto_name"]).set_index("canonical_key")["proto_name"].astype(str)
    family_by_key = assignment.map(STRUCTURAL_FAMILY)
    row_family = raw["canonical_key"].map(family_by_key)
    y = pd.to_numeric(raw["Value"], errors="raise").astype(float)
    predictions: list[pd.DataFrame] = []
    params = dict(CB_PARAMS)
    params["thread_count"] = 1
    for family_index, family in enumerate(["layered", "olivine", "spinel"]):
        test_mask = row_family.eq(family).to_numpy()
        train_mask = ~test_mask
        if not test_mask.any():
            continue
        params["random_seed"] = 43000 + family_index
        model = CatBoostRegressor(**params)
        model.fit(features.loc[train_mask], y.loc[train_mask])
        predicted = np.asarray(model.predict(features.loc[test_mask], thread_count=1), dtype=float)
        part = pd.DataFrame(
            {
                "row_id": raw.loc[test_mask, "row_id"].to_numpy(int),
                "canonical_key": raw.loc[test_mask, "canonical_key"].astype(str).to_numpy(),
                "held_out_family": family,
                "y_true": y.loc[test_mask].to_numpy(float),
                "y_pred": predicted,
            }
        )
        predictions.append(part)
        log(f"Leave-one-family-out model complete: {family}")
    combined = pd.concat(predictions, ignore_index=True).sort_values("row_id").reset_index(drop=True)
    combined["abs_error"] = np.abs(combined["y_true"] - combined["y_pred"])
    combined.to_csv(RESULTS / "leave_one_family_out_predictions.csv", index=False)
    rows = []
    for label, part in [("overall", combined), *combined.groupby("held_out_family", sort=True)]:
        metrics = point_metrics(part["y_true"].to_numpy(float), part["y_pred"].to_numpy(float))
        rows.append(
            {
                "scope": label,
                "n_records": int(len(part)),
                "n_canonical_compositions": int(part["canonical_key"].nunique()),
                **metrics,
            }
        )
    summary_frame = pd.DataFrame(rows)
    summary_frame.to_csv(RESULTS / "leave_one_family_out_summary.csv", index=False)
    overall = summary_frame[summary_frame["scope"].eq("overall")].iloc[0]
    overall_ci = clustered_metric_ci(combined, seed=42019)
    output = {
        "definition": "all records assigned to one structural family are excluded from training; unassigned records remain in training",
        "families": ["layered", "olivine", "spinel"],
        "overall": {
            **{key: (int(overall[key]) if key.startswith("n_") else float(overall[key])) for key in ["n_records", "n_canonical_compositions", "r2", "mae", "rmse", "spearman"]},
            "cluster_bootstrap_ci95": overall_ci,
            "bootstrap_unit": "canonical_composition",
            "bootstrap_repetitions": BOOTSTRAP_REPS,
        },
        "by_family": json.loads(
            summary_frame[~summary_frame["scope"].eq("overall")].to_json(orient="records")
        ),
    }
    (RESULTS / "leave_one_family_out_validation.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def load_reference_arm() -> tuple[str, float, float]:
    decision = json.loads((RESULTS / "nsga_decision.json").read_text(encoding="utf-8"))
    arm = "nsga2__floor_on" if decision["retain_nsga2_as_substantive_performance_claim"] else "random__floor_on"
    known = json.loads((RESULTS / "known_control_summary.json").read_text(encoding="utf-8"))
    return (
        arm,
        float(known["positive_control_radius_p90_after_g1_g2"]),
        float(known["positive_control_radius_p95_after_g1_g2"]),
    )


def candidate_recurrence() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    arm, p90, p95 = load_reference_arm()
    frames = []
    for seed in SEEDS:
        frame = pd.read_csv(SEARCH / f"{arm}_seed{seed}_topk_screened.csv").copy()
        frame["seed"] = seed
        frame["pass_conservative_p90_final"] = (
            frame["g1_pass"].astype(bool)
            & frame["g2_pass"].astype(bool)
            & frame["g4_pass"].astype(bool)
            & pd.to_numeric(frame["radius_mismatch"], errors="coerce").le(p90)
        )
        frame["pass_reference_p95_final"] = (
            frame["g1_pass"].astype(bool)
            & frame["g2_pass"].astype(bool)
            & frame["g4_pass"].astype(bool)
            & pd.to_numeric(frame["radius_mismatch"], errors="coerce").le(p95)
        )
        frames.append(frame)
    all_rows = pd.concat(frames, ignore_index=True)
    records = []
    for canonical_key, part in all_rows.groupby("canonical_key", sort=True):
        first = part.sort_values(["seed", "pareto_rank", "canonical_key"], kind="mergesort").iloc[0]
        seeds = sorted(part["seed"].astype(int).unique())
        p90_seeds = sorted(part.loc[part["pass_conservative_p90_final"], "seed"].astype(int).unique())
        p95_seeds = sorted(part.loc[part["pass_reference_p95_final"], "seed"].astype(int).unique())
        records.append(
            {
                "canonical_key": canonical_key,
                "Formula": first["Formula"],
                "proto_name": first["proto_name"],
                "topk_seed_count": len(seeds),
                "topk_seed_fraction": len(seeds) / len(SEEDS),
                "p90_final_seed_count": len(p90_seeds),
                "p95_final_seed_count": len(p95_seeds),
                "passes_p90_every_occurrence": len(p90_seeds) == len(seeds),
                "passes_p95_every_occurrence": len(p95_seeds) == len(seeds),
                "topk_seeds": ";".join(map(str, seeds)),
                "p90_final_seeds": ";".join(map(str, p90_seeds)),
                "p95_final_seeds": ";".join(map(str, p95_seeds)),
                "y_pred_mean_across_occurrences": float(part["y_pred"].mean()),
                "y_pred_sd_across_occurrences": float(part["y_pred"].std(ddof=1)) if len(part) > 1 else 0.0,
                "y_discounted_mean_across_occurrences": float(part["y_discounted"].mean()),
                "model_distance_mean_across_occurrences": float(part["model_distance"].mean()),
                "radius_mismatch": float(pd.to_numeric(part["radius_mismatch"]).median()),
                "q_cationic_redox": float(pd.to_numeric(part["q_cationic_redox"]).median()),
                "recurrence_consensus_10_of_20": len(seeds) >= 10,
                "primary_consensus_candidate": (
                    len(seeds) >= 10
                    and len(p95_seeds) == len(seeds)
                    and bool(part["g1_pass"].astype(bool).all())
                    and bool(part["g2_pass"].astype(bool).all())
                    and bool(part["g4_pass"].astype(bool).all())
                ),
            }
        )
    recurrence = pd.DataFrame(records).sort_values(
        ["topk_seed_count", "p90_final_seed_count", "y_discounted_mean_across_occurrences"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    consensus = recurrence[recurrence["primary_consensus_candidate"]].copy().reset_index(drop=True)
    recurrence.to_csv(RESULTS / "candidate_recurrence_union.csv", index=False)
    consensus.to_csv(RESULTS / "consensus_candidates.csv", index=False)
    output = {
        "reference_arm": arm,
        "seeds": SEEDS,
        "recurrence_rule": "present in at least 10 of 20 independent matched-search seeds",
        "primary_scientific_shortlist_rule": "recurrence rule plus G1, G2, G4 and control-P95 G3 pass in every occurrence; P90 is a nested conservative flag",
        "union_candidates": int(len(recurrence)),
        "recurrence_consensus_candidates": int(recurrence["recurrence_consensus_10_of_20"].sum()),
        "primary_consensus_candidates": int(len(consensus)),
        "p90_threshold": p90,
        "p95_threshold": p95,
    }
    (RESULTS / "candidate_recurrence_summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return recurrence, consensus, output


def candidate_features(recurrence: pd.DataFrame, evaluator: CandidateEvaluator) -> pd.DataFrame:
    candidates = []
    for row in recurrence.itertuples(index=False):
        comp = Composition(str(row.Formula))
        candidates.append(
            {
                "comp": comp.get_el_amt_dict(),
                "Formula": str(row.Formula),
                "canonical_key": str(row.canonical_key),
                "proto_name": str(row.proto_name),
                "is_doped": 1,
            }
        )
    rows = [evaluator._composition_row(candidate) for candidate in candidates]
    return pd.DataFrame(rows).reindex(columns=evaluator.comp_columns)


def element_support(recurrence: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_csv(DATA / "clean_rows.csv")
    unique = raw.drop_duplicates("canonical_key")
    records = []
    for element in PERIODIC_ELEMENTS:
        unique_count = int(unique["canonical_formula"].map(lambda value: float(Composition(value).get(element, 0)) > 0).sum())
        record_count = int(raw["canonical_formula"].map(lambda value: float(Composition(value).get(element, 0)) > 0).sum())
        candidate_count = int(recurrence["Formula"].map(lambda value: float(Composition(value).get(element, 0)) > 0).sum())
        if unique_count or candidate_count:
            records.append(
                {
                    "element": element,
                    "training_canonical_compositions": unique_count,
                    "training_records": record_count,
                    "candidate_union_compositions": candidate_count,
                }
            )
    support = pd.DataFrame(records).sort_values(["training_canonical_compositions", "element"])
    support.to_csv(RESULTS / "element_support_summary.csv", index=False)
    return support


def candidate_uncertainty_and_context(
    recurrence: pd.DataFrame,
    consensus: pd.DataFrame,
) -> dict[str, object]:
    evaluator = CandidateEvaluator()
    X_comp = candidate_features(recurrence, evaluator)
    raw = pd.read_csv(DATA / "clean_rows.csv")
    X_train = pd.read_csv(DATA / "features_composition_context.csv")
    y = pd.to_numeric(raw["Value"], errors="raise").astype(float)
    groups = raw["canonical_key"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique_groups}

    nominal = X_comp.copy()
    nominal["temperature_K"] = 298.15
    nominal["log1p_current_rate_C"] = math.log1p(0.5)
    nominal["log1p_cycle_number"] = math.log1p(1.0)
    nominal["log1p_current_rate_x_cycle"] = math.log1p(0.5)
    nominal = nominal.reindex(columns=evaluator.feature_columns)

    rng = np.random.default_rng(42031)
    bootstrap_predictions = np.empty((len(recurrence), MODEL_BOOTSTRAP_REPS), dtype=float)
    base_params = dict(CB_PARAMS)
    base_params["thread_count"] = 1
    for repetition in range(MODEL_BOOTSTRAP_REPS):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([by_group[group] for group in sampled_groups])
        params = dict(base_params)
        params["random_seed"] = 44000 + repetition
        model = CatBoostRegressor(**params)
        model.fit(X_train.iloc[indices], y.iloc[indices])
        bootstrap_predictions[:, repetition] = np.asarray(model.predict(nominal, thread_count=1), dtype=float)
        if (repetition + 1) % 10 == 0:
            log(f"Candidate model bootstrap {repetition + 1}/{MODEL_BOOTSTRAP_REPS}")

    uncertainty = recurrence[["canonical_key", "Formula", "proto_name", "topk_seed_count", "primary_consensus_candidate", "radius_mismatch"]].copy()
    uncertainty["bootstrap_prediction_mean"] = bootstrap_predictions.mean(axis=1)
    uncertainty["bootstrap_prediction_sd"] = bootstrap_predictions.std(axis=1, ddof=1)
    uncertainty["bootstrap_prediction_p05"] = np.percentile(bootstrap_predictions, 5, axis=1)
    uncertainty["bootstrap_prediction_p50"] = np.percentile(bootstrap_predictions, 50, axis=1)
    uncertainty["bootstrap_prediction_p95"] = np.percentile(bootstrap_predictions, 95, axis=1)
    uncertainty["bootstrap_prediction_90_interval_width"] = uncertainty["bootstrap_prediction_p95"] - uncertainty["bootstrap_prediction_p05"]

    thresholds = pd.read_csv(RESULTS / "control_threshold_bootstrap.csv")
    p90_draws = thresholds.loc[thresholds["policy"].eq("P90"), "threshold"].to_numpy(float)
    p95_draws = thresholds.loc[thresholds["policy"].eq("P95"), "threshold"].to_numpy(float)
    radius = uncertainty["radius_mismatch"].to_numpy(float)
    uncertainty["p90_threshold_pass_probability"] = (radius[:, None] <= p90_draws[None, :]).mean(axis=1)
    uncertainty["p95_threshold_pass_probability"] = (radius[:, None] <= p95_draws[None, :]).mean(axis=1)
    uncertainty["threshold_robust_reference_candidate"] = (
        uncertainty["primary_consensus_candidate"].astype(bool)
        & uncertainty["p95_threshold_pass_probability"].ge(0.95)
    )
    uncertainty["below_bootstrap_p05_120"] = uncertainty["bootstrap_prediction_p05"] >= 120.0
    uncertainty.to_csv(RESULTS / "candidate_model_uncertainty.csv", index=False)

    full_model = joblib.load(MODELS / "catboost_full_context.joblib")
    context_rows = recurrence[["canonical_key", "Formula", "proto_name", "topk_seed_count", "primary_consensus_candidate"]].copy()
    prediction_columns = []
    for rate in [0.1, 0.5, 1.0, 5.0]:
        for cycle in [1.0, 50.0, 100.0, 200.0]:
            X = X_comp.copy()
            X["temperature_K"] = 298.15
            X["log1p_current_rate_C"] = math.log1p(rate)
            X["log1p_cycle_number"] = math.log1p(cycle)
            X["log1p_current_rate_x_cycle"] = math.log1p(rate * cycle)
            X = X.reindex(columns=evaluator.feature_columns)
            column = f"pred_{rate:g}C_cycle{cycle:g}"
            context_rows[column] = np.asarray(full_model.predict(X, thread_count=1), dtype=float)
            prediction_columns.append(column)
    context_rows["context_prediction_mean"] = context_rows[prediction_columns].mean(axis=1)
    context_rows["context_prediction_min"] = context_rows[prediction_columns].min(axis=1)
    context_rows["context_prediction_max"] = context_rows[prediction_columns].max(axis=1)
    context_rows["context_prediction_range"] = context_rows["context_prediction_max"] - context_rows["context_prediction_min"]
    context_rows["context_min_ge_120"] = context_rows["context_prediction_min"] >= 120.0
    context_rows.to_csv(RESULTS / "candidate_context_robustness.csv", index=False)

    support = element_support(recurrence)
    consensus_uncertainty = uncertainty[uncertainty["primary_consensus_candidate"]].copy()
    consensus_context = context_rows[context_rows["primary_consensus_candidate"]].copy()
    output = {
        "model_bootstrap_repetitions": MODEL_BOOTSTRAP_REPS,
        "model_bootstrap_unit": "canonical_composition",
        "model_prediction_context": {"temperature_K": 298.15, "current_rate_C": 0.5, "cycle": 1},
        "context_grid": {"temperature_K": [298.15], "current_rate_C": [0.1, 0.5, 1.0, 5.0], "cycle": [1, 50, 100, 200]},
        "candidate_union_n": int(len(recurrence)),
        "primary_consensus_n": int(len(consensus)),
        "threshold_robust_reference_n": int(
            uncertainty["threshold_robust_reference_candidate"].sum()
        ),
        "consensus_model_interval_width_range": [
            float(consensus_uncertainty["bootstrap_prediction_90_interval_width"].min()) if len(consensus_uncertainty) else np.nan,
            float(consensus_uncertainty["bootstrap_prediction_90_interval_width"].max()) if len(consensus_uncertainty) else np.nan,
        ],
        "consensus_context_minimum_range": [
            float(consensus_context["context_prediction_min"].min()) if len(consensus_context) else np.nan,
            float(consensus_context["context_prediction_min"].max()) if len(consensus_context) else np.nan,
        ],
        "training_support_for_elements_in_consensus": support[
            support["element"].isin(
                sorted({element.symbol for formula in consensus["Formula"] for element in Composition(formula).elements})
            )
        ].to_dict(orient="records"),
    }
    (RESULTS / "candidate_uncertainty_summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def write_validation_hierarchy(
    nominal: dict[str, object],
    family: dict[str, object],
) -> None:
    results = json.loads((RESULTS / "results_for_manuscript.json").read_text(encoding="utf-8"))
    all_oof = pd.read_csv(RESULTS / "oof_canonical_composition_heldout_all_models.csv")
    exact_oof = all_oof[all_oof["model"].eq("catboost__composition_plus_context")][
        ["canonical_key", "y_true", "y_pred"]
    ].copy()
    exact_ci = clustered_metric_ci(exact_oof, seed=42017)
    record = results["record_cv_catboost_context"]
    exact = results["canonical_composition_heldout_catboost_context"]
    rows = [
        {"validation_level": "record-level CV", "r2": record["r2"], "r2_ci95_low": record["cluster_bootstrap_ci95"]["r2"][0], "r2_ci95_high": record["cluster_bootstrap_ci95"]["r2"][1], "n_records": 907, "n_units": 907},
        {"validation_level": "exact canonical-composition holdout", "r2": exact["r2"], "r2_ci95_low": exact_ci["r2"][0], "r2_ci95_high": exact_ci["r2"][1], "n_records": 907, "n_units": results["data"]["canonical_compositions_decimals_8"]},
        {"validation_level": "nominal-context slice within canonical holdout", "r2": nominal["r2"], "r2_ci95_low": nominal["cluster_bootstrap_ci95"]["r2"][0], "r2_ci95_high": nominal["cluster_bootstrap_ci95"]["r2"][1], "n_records": nominal["n_records"], "n_units": nominal["n_canonical_compositions"]},
        {"validation_level": "leave-one-structural-family-out", "r2": family["overall"]["r2"], "r2_ci95_low": family["overall"]["cluster_bootstrap_ci95"]["r2"][0], "r2_ci95_high": family["overall"]["cluster_bootstrap_ci95"]["r2"][1], "n_records": family["overall"]["n_records"], "n_units": family["overall"]["n_canonical_compositions"]},
    ]
    pd.DataFrame(rows).to_csv(RESULTS / "validation_hierarchy.csv", index=False)


def main() -> None:
    log("Evaluating the exact nominal-context slice")
    nominal = nominal_context_validation()
    log("Running leave-one-structural-family-out validation")
    family = leave_one_family_out_validation()
    write_validation_hierarchy(nominal, family)
    log("Calculating 20-seed candidate recurrence")
    recurrence, consensus, recurrence_summary = candidate_recurrence()
    log("Running candidate model-resampling and context robustness")
    uncertainty = candidate_uncertainty_and_context(recurrence, consensus)
    combined = {
        "nominal_context": nominal,
        "leave_one_family_out": family,
        "candidate_recurrence": recurrence_summary,
        "candidate_uncertainty": uncertainty,
    }
    (RESULTS / "extended_validation_summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    log("Extended validation complete")


if __name__ == "__main__":
    main()
