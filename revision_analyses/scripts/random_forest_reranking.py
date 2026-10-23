"""Re-rank the saved candidate pools with a Random Forest full model.

The search itself was guided by CatBoost; this check re-ranks the saved pools
only, it does not repeat the search.

1. Fits a Random Forest full model on the same 907 records and 208
   composition + context features as the CatBoost full model, with the
   manuscript's RF settings (rebuild_data_and_models.model_factories()
   ["random_forest"], used unchanged), and saves it.
2. Predicts every composition in the 40 saved candidate pools at the fixed
   context (298.15 K, 0.5 C, cycle 1), clipped to the same training range
   as the CatBoost evaluator.  model_distance, discount and comp_novelty are
   kept from the pools (they do not depend on the surrogate), so
   y_discounted = RF prediction x saved discount.
3. Re-selects TopK with the original rule, re-screens G1-G4 (P95, G4 against
   the RF prediction) and recomputes recurrence for the reference arm.
Outputs: outputs/random_forest_reranking/ (model, features cache, TopK, summary).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pymatgen.core import Composition

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1] / "paper_release_v1"
SAVED = HERE.parents[1] / "paper_release_v1"
OUT = HERE.parent / "outputs" / "random_forest_reranking"
SEEDS = list(range(42, 62))
sys.path.insert(0, str(WORK / "src"))

import search_screen_benchmark as ssb  # noqa: E402
from rebuild_data_and_models import model_factories  # noqa: E402
from postprocess_final import jaccard  # noqa: E402

START = time.time()


def log(message: str) -> None:
    print(f"[{time.time() - START:7.1f}s] {message}", flush=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "search").mkdir(exist_ok=True)

    # 1. Random Forest full model on the saved training table.
    clean = pd.read_csv(SAVED / "data" / "clean_rows.csv")
    X_train = pd.read_csv(SAVED / "data" / "features_composition_context.csv")
    y = pd.to_numeric(clean["Value"], errors="raise").astype(float)
    rf = model_factories()["random_forest"]()
    rf.fit(X_train, y)
    local = HERE.parent / "models_local"
    local.mkdir(parents=True, exist_ok=True)
    model_path = local / "random_forest_full_context.joblib"
    joblib.dump(rf, model_path)
    log("RF full model fitted and saved")

    # 2. Features for every unique composition in the 40 pools (cached).
    evaluator = ssb.CandidateEvaluator()
    pools = {}
    for seed in SEEDS:
        for proposal in ["nsga2", "random"]:
            pools[(proposal, seed)] = ssb.read_checkpoint_csv(
                SAVED / "results" / "search_candidates" / f"{proposal}_seed{seed}_candidate_pool.csv.gz")
    unique = (pd.concat([p[["canonical_key", "Formula"]] for p in pools.values()])
              .drop_duplicates("canonical_key").sort_values("canonical_key").reset_index(drop=True))
    cache = local / "pool_features_composition.npz"
    if cache.exists():
        stored = np.load(cache, allow_pickle=False)
        if list(stored["keys"]) != list(unique["canonical_key"]):
            raise RuntimeError("Feature cache does not match the pools")
        X_comp = pd.DataFrame(stored["values"], columns=list(stored["columns"]))
    else:
        # Fill a preallocated array and empty the evaluator's per-composition cache as we go,
        # so 375k feature dictionaries are never held in memory at once.
        columns = list(evaluator.comp_columns)
        values = np.full((len(unique), len(columns)), np.nan, dtype=float)
        for i, (key, formula) in enumerate(zip(unique["canonical_key"], unique["Formula"])):
            row = evaluator._composition_row({"comp": Composition(formula).get_el_amt_dict(),
                                              "Formula": formula, "canonical_key": key})
            values[i] = [row.get(column, np.nan) for column in columns]
            if (i + 1) % 10000 == 0:
                evaluator.cache.clear()
            if (i + 1) % 50000 == 0:
                log(f"featurized {i + 1}/{len(unique)}")
        evaluator.cache.clear()
        X_comp = pd.DataFrame(values, columns=columns)
        np.savez_compressed(cache, keys=unique["canonical_key"].to_numpy(str),
                            columns=np.array(evaluator.comp_columns, dtype=str), values=X_comp.to_numpy(float))
    log(f"features ready for {len(unique)} compositions")
    X = X_comp.copy()
    X["temperature_K"] = ssb.FIXED_TEMPERATURE_K
    X["log1p_current_rate_C"] = np.log1p(ssb.FIXED_CURRENT_RATE_C)
    X["log1p_cycle_number"] = np.log1p(ssb.FIXED_CYCLE)
    X["log1p_current_rate_x_cycle"] = np.log1p(ssb.FIXED_CURRENT_RATE_C * ssb.FIXED_CYCLE)
    X = X.reindex(columns=evaluator.feature_columns)
    rf_unclipped = np.asarray(rf.predict(X), dtype=float)
    rf_pred = np.clip(rf_unclipped, *evaluator.prediction_clip)
    prediction = pd.DataFrame({"canonical_key": unique["canonical_key"], "rf_pred_unclipped": rf_unclipped,
                               "rf_pred": rf_pred})
    prediction.to_csv(OUT / "rf_pool_predictions.csv", index=False)
    log("RF predictions done")
    lookup_unclipped = dict(zip(prediction["canonical_key"], prediction["rf_pred_unclipped"]))
    lookup = dict(zip(prediction["canonical_key"], prediction["rf_pred"]))

    # 3. Re-rank, re-screen and recurrence.
    known = json.loads((SAVED / "results" / "known_control_summary.json").read_text(encoding="utf-8"))
    p90 = float(known["positive_control_radius_p90_after_g1_g2"])
    p95 = float(known["positive_control_radius_p95_after_g1_g2"])
    summaries = []
    for (proposal, seed), pool in pools.items():
        pool = pool.copy()
        pool["y_pred_catboost"] = pool["y_pred"]
        pool["y_pred_unclipped"] = pool["canonical_key"].map(lookup_unclipped)
        pool["y_pred"] = pool["canonical_key"].map(lookup)
        pool["y_discounted"] = pool["y_pred"] * pool["discount"]
        for floor_on in [True, False]:
            arm = f"{proposal}__floor_{'on' if floor_on else 'off'}"
            topk = ssb.select_topk(pool, floor_on=floor_on)
            screened = ssb.screen(topk, radius_threshold=p95)
            screened["g3_p90_pass"] = screened["pass_g1_g2"] & screened["radius_mismatch"].le(p90)
            screened["g3_p95_pass"] = screened["pass_g1_g2"] & screened["radius_mismatch"].le(p95)
            screened["seed"], screened["arm"] = seed, arm
            screened.to_csv(OUT / "search" / f"{arm}_seed{seed}_topk_screened_rf.csv", index=False)
            summaries.append(ssb.arm_summary(topk, screened, seed, proposal, floor_on))
    summary = pd.DataFrame(summaries).sort_values(["seed", "arm"])
    summary.to_csv(OUT / "rf_search_seed_summary.csv", index=False)
    log("re-ranking and screening done")

    frames = []
    for seed in SEEDS:
        frame = pd.read_csv(OUT / "search" / f"nsga2__floor_on_seed{seed}_topk_screened_rf.csv")
        base = frame["g1_pass"].astype(bool) & frame["g2_pass"].astype(bool) & frame["g4_pass"].astype(bool)
        frame["pass_reference_p95_final"] = base & pd.to_numeric(frame["radius_mismatch"], errors="coerce").le(p95)
        frames.append(frame)
    rows = pd.concat(frames, ignore_index=True)
    records = []
    for key, part in rows.groupby("canonical_key", sort=True):
        seeds = sorted(part["seed"].astype(int).unique())
        passing = sorted(part.loc[part["pass_reference_p95_final"], "seed"].astype(int).unique())
        records.append({
            "canonical_key": key, "Formula": part["Formula"].iloc[0], "proto_name": part["proto_name"].iloc[0],
            "topk_seed_count": len(seeds), "p95_final_seed_count": len(passing),
            "rf_pred": float(part["y_pred"].mean()), "catboost_pred": float(part["y_pred_catboost"].mean()),
            "consensus_rf": (len(seeds) >= 10 and len(passing) == len(seeds)
                             and bool(part["g1_pass"].astype(bool).all()) and bool(part["g2_pass"].astype(bool).all())
                             and bool(part["g4_pass"].astype(bool).all())),
        })
    union = pd.DataFrame(records).sort_values("topk_seed_count", ascending=False)
    union.to_csv(OUT / "rf_recurrence_union.csv", index=False)
    consensus = union[union["consensus_rf"]]
    original = pd.read_csv(SAVED / "results" / "consensus_candidates.csv")
    original_counts = union.set_index("Formula").reindex(original["Formula"])

    seven = original[["Formula"]].copy()
    seven["catboost_pred_saved"] = original["y_pred_mean_across_occurrences"].to_numpy()
    seven["rf_pred"] = [lookup[k] for k in original["canonical_key"]]
    seven["rf_pred_unclipped"] = [lookup_unclipped[k] for k in original["canonical_key"]]
    seven["catboost_seed_count"] = original["topk_seed_count"].to_numpy()
    seven["rf_topk_seed_count"] = original_counts["topk_seed_count"].fillna(0).astype(int).to_numpy()
    seven["rf_p95_final_seed_count"] = original_counts["p95_final_seed_count"].fillna(0).astype(int).to_numpy()
    seven["rf_consensus"] = original_counts["consensus_rf"].fillna(False).astype(bool).to_numpy()
    seven.to_csv(OUT / "rf_seven_candidates.csv", index=False)

    agg = summary.groupby("arm")[["g4_survivors", "feasible_hypervolume", "prototype_coverage"]].mean()
    result = {
        "rf_model": {"file": model_path.name, "sha256": sha256(model_path),
                     "params": {k: str(v) for k, v in rf.get_params().items() if k.startswith("model__")},
                     "training_rows": int(len(y)), "features": int(X_train.shape[1])},
        "prediction_clip": evaluator.prediction_clip,
        "unique_pool_compositions": int(len(unique)),
        "reference_topk_union": int(len(union)),
        "reference_topk_mean_pairwise_jaccard": jaccard([set(f["canonical_key"]) for f in frames]),
        "candidates_with_10plus_seeds": int((union["topk_seed_count"] >= 10).sum()),
        "rf_consensus": consensus[["Formula", "proto_name", "topk_seed_count", "rf_pred", "catboost_pred"]].to_dict(orient="records"),
        "original_retained": sorted(set(original["Formula"]) & set(consensus["Formula"])),
        "new_recurrent": sorted(set(consensus["Formula"]) - set(original["Formula"])),
        "per_arm_means": agg.reset_index().to_dict(orient="records"),
        "runtime_seconds": round(time.time() - START, 1),
    }
    (OUT / "rf_ranking_summary.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(json.dumps(result, indent=1, default=float))
    print(seven.round(2).to_string())
    print(union.head(15).round(2).to_string())


if __name__ == "__main__":
    main()
