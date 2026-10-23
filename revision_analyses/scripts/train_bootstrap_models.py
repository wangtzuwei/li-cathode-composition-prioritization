"""The 100 composition-bootstrap CatBoost models of the release, regenerated and saved.

Repeats the bootstrap loop of extended_validation.candidate_uncertainty_and_context()
statement for statement (same generator default_rng(42031), same group draw,
CB_PARAMS with thread_count=1 and random_seed=44000+rep), and additionally
saves every fitted model with its seed and sampling indices, so that new
comparison compositions can later be scored with the very same 100 models.

Inputs:
  - paper_release_v1/data/,
  - the saved candidate union (results/candidate_recurrence_union.csv) and the
    saved G3 threshold bootstrap (results/control_threshold_bootstrap.csv).
Outputs (models_local/; the release files are not modified):
  - bootstrap_models/candidate_model_uncertainty.csv, candidate_uncertainty_summary.json  (same columns as the release)
  - bootstrap_models/model_000.cbm ... model_099.cbm
  - bootstrap_models/sampling_indices.npz   (row indices and group indices per model)
  - bootstrap_models/predictions_union.npy  (9,815 x 100 nominal-context predictions)
  - bootstrap_models/manifest.json           (seeds, parameters, data and model SHA-256)
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1] / "paper_release_v1"
SAVED = HERE.parents[1] / "paper_release_v1"
MODEL_DIR = HERE.parent / "models_local" / "bootstrap_models"
sys.path.insert(0, str(WORK / "src"))

from catboost import CatBoostRegressor  # noqa: E402

import extended_validation as ev  # noqa: E402
from rebuild_data_and_models import CB_PARAMS  # noqa: E402
from search_screen_benchmark import CandidateEvaluator  # noqa: E402

RNG_SEED = 42031
REPS = ev.MODEL_BOOTSTRAP_REPS  # 100


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    start = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    recurrence = pd.read_csv(SAVED / "results" / "candidate_recurrence_union.csv")

    evaluator = CandidateEvaluator()
    X_comp = ev.candidate_features(recurrence, evaluator)
    raw = pd.read_csv(ev.DATA / "clean_rows.csv")
    X_train = pd.read_csv(ev.DATA / "features_composition_context.csv")
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

    rng = np.random.default_rng(RNG_SEED)
    predictions = np.empty((len(recurrence), REPS), dtype=float)
    base_params = dict(CB_PARAMS)
    base_params["thread_count"] = 1
    index_arrays: dict[str, np.ndarray] = {}
    models = []
    group_position = {group: i for i, group in enumerate(unique_groups)}
    for repetition in range(REPS):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([by_group[group] for group in sampled_groups])
        params = dict(base_params)
        params["random_seed"] = 44000 + repetition
        model = CatBoostRegressor(**params)
        model.fit(X_train.iloc[indices], y.iloc[indices])
        predictions[:, repetition] = np.asarray(model.predict(nominal, thread_count=1), dtype=float)

        path = MODEL_DIR / f"model_{repetition:03d}.cbm"
        model.save_model(str(path))
        index_arrays[f"rows_{repetition:03d}"] = indices.astype(np.int32)
        index_arrays[f"groups_{repetition:03d}"] = np.array([group_position[g] for g in sampled_groups], dtype=np.int32)
        models.append({"repetition": repetition, "random_seed": 44000 + repetition, "file": path.name,
                       "sha256": sha256(path), "n_training_rows": int(len(indices))})
        if (repetition + 1) % 10 == 0:
            print(f"[{time.time() - start:7.1f}s] bootstrap {repetition + 1}/{REPS}", flush=True)

    np.savez_compressed(MODEL_DIR / "sampling_indices.npz", unique_groups=unique_groups.astype(str), **index_arrays)
    np.save(MODEL_DIR / "predictions_union.npy", predictions)

    # Same table construction as candidate_uncertainty_and_context().
    uncertainty = recurrence[["canonical_key", "Formula", "proto_name", "topk_seed_count",
                              "primary_consensus_candidate", "radius_mismatch"]].copy()
    uncertainty["bootstrap_prediction_mean"] = predictions.mean(axis=1)
    uncertainty["bootstrap_prediction_sd"] = predictions.std(axis=1, ddof=1)
    uncertainty["bootstrap_prediction_p05"] = np.percentile(predictions, 5, axis=1)
    uncertainty["bootstrap_prediction_p50"] = np.percentile(predictions, 50, axis=1)
    uncertainty["bootstrap_prediction_p95"] = np.percentile(predictions, 95, axis=1)
    uncertainty["bootstrap_prediction_90_interval_width"] = (
        uncertainty["bootstrap_prediction_p95"] - uncertainty["bootstrap_prediction_p05"])
    thresholds = pd.read_csv(SAVED / "results" / "control_threshold_bootstrap.csv")
    p90_draws = thresholds.loc[thresholds["policy"].eq("P90"), "threshold"].to_numpy(float)
    p95_draws = thresholds.loc[thresholds["policy"].eq("P95"), "threshold"].to_numpy(float)
    radius = uncertainty["radius_mismatch"].to_numpy(float)
    uncertainty["p90_threshold_pass_probability"] = (radius[:, None] <= p90_draws[None, :]).mean(axis=1)
    uncertainty["p95_threshold_pass_probability"] = (radius[:, None] <= p95_draws[None, :]).mean(axis=1)
    uncertainty["threshold_robust_reference_candidate"] = (
        uncertainty["primary_consensus_candidate"].astype(bool) & uncertainty["p95_threshold_pass_probability"].ge(0.95))
    uncertainty["below_bootstrap_p05_120"] = uncertainty["bootstrap_prediction_p05"] >= 120.0
    uncertainty.to_csv(MODEL_DIR / "candidate_model_uncertainty.csv", index=False)

    consensus = uncertainty[uncertainty["primary_consensus_candidate"].astype(bool)]
    manifest = {
        "purpose": "100 composition-bootstrap CatBoost models of the submitted analysis, retrained 2026-10-22",
        "procedure": "extended_validation.candidate_uncertainty_and_context bootstrap loop, unchanged",
        "generator": f"numpy.random.default_rng({RNG_SEED}); groups drawn with rng.choice(unique_groups, size=n_groups, replace=True)",
        "bootstrap_unit": "canonical_composition",
        "catboost_params": {k: v for k, v in base_params.items()},
        "random_seed_rule": "44000 + repetition",
        "prediction_context": {"temperature_K": 298.15, "current_rate_C": 0.5, "cycle": 1},
        "training_data_sha256": {
            "clean_rows.csv": sha256(ev.DATA / "clean_rows.csv"),
            "features_composition_context.csv": sha256(ev.DATA / "features_composition_context.csv"),
        },
        "feature_columns": list(evaluator.feature_columns),
        "how_to_score_new_compositions": (
            "Build the 208 features with search_screen_benchmark.CandidateEvaluator._composition_row "
            "plus the four context columns, reindex to feature_columns, and predict with each model_XXX.cbm "
            "(CatBoostRegressor().load_model, thread_count=1); take the 5th/95th percentile over the 100 predictions."),
        "sampling_indices": "sampling_indices.npz: rows_XXX = training row positions (0-based, clean_rows order), "
                            "groups_XXX = positions in unique_groups",
        "consensus_interval_width_range": [float(consensus["bootstrap_prediction_90_interval_width"].min()),
                                           float(consensus["bootstrap_prediction_90_interval_width"].max())],
        "models": models,
        "runtime_seconds": round(time.time() - start, 1),
    }
    (MODEL_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Only the bootstrap-dependent part of candidate_uncertainty_summary.json (read by the ledger).
    (MODEL_DIR / "candidate_uncertainty_summary.json").write_text(json.dumps({
        "consensus_model_interval_width_range": manifest["consensus_interval_width_range"],
    }, indent=2), encoding="utf-8")
    print(f"done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
