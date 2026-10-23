"""23: model random_seed and the ranking (analysis 23; see revision_analyses/README.md).

1. Trains 5 CatBoost full models with the release settings (rebuild_data_and_models.CB_PARAMS) on the same
   907 records and 208 composition + context features, changing only random_seed (1-5).  Saved to
   models_local/model_seed_robustness/ (nothing existing is overwritten).
2. Predicts every composition of the 20 saved NSGA-II candidate pools at the fixed search context with the
   feature matrix cached by the random-forest re-ranking (models_local/pool_features_composition.npz), clipped to the evaluator's range.
   model_distance, discount and comp_novelty do not depend on the surrogate and are kept from the pools, so
   y_discounted = new prediction x saved discount.
3. Re-selects TopK (reference arm nsga2__floor_on), re-screens G1-G4 at the saved pooled P95 (G4 against the
   new prediction), and recomputes formula-level recurrence (>= 10 seeds and passing in every seed recovered)
   and motif-level recurrence (candidate_checks.motif_table).
Check: the same pipeline run with the saved full model (random_seed 42) must reproduce the saved TopK sets.
No search is run.  Outputs: outputs/23_*.
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

import candidate_checks as t
from candidate_checks import HERE, OUT, SAVED, SEEDS, SEVEN, log, ssb, write_json
from rebuild_data_and_models import CB_PARAMS  # noqa: E402 - path set by candidate_checks

MODEL_SEEDS = [1, 2, 3, 4, 5]
MODEL_DIR = HERE.parent / "models_local" / "model_seed_robustness"
FEATURES = HERE.parent / "models_local" / "pool_features_composition.npz"
TARGET_MOTIF = "Li2MnO3 | Cr+F"
LI2 = "Li2Mn0.7Cr0.3O2.865F0.135"


def train_models() -> dict[int, CatBoostRegressor]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    clean = pd.read_csv(SAVED / "data" / "clean_rows.csv")
    X = pd.read_csv(SAVED / "data" / "features_composition_context.csv")
    y = pd.to_numeric(clean["Value"], errors="raise").astype(float)
    models, meta = {}, []
    for seed in MODEL_SEEDS:
        path = MODEL_DIR / f"catboost_full_context_seed{seed}.cbm"
        if path.exists():
            model = CatBoostRegressor()
            model.load_model(str(path))
        else:
            model = CatBoostRegressor(**dict(CB_PARAMS, random_seed=seed))
            model.fit(X, y)
            model.save_model(str(path))
        models[seed] = model
        meta.append({"random_seed": seed, "file": path.name,
                     "train_r2": float(1 - np.mean((model.predict(X) - y) ** 2) / np.var(y))})
    (MODEL_DIR / "manifest.json").write_text(json.dumps({
        "params": {k: (str(v) if k != "random_seed" else "per model") for k, v in CB_PARAMS.items()},
        "training_rows": int(len(y)), "feature_columns": list(X.columns), "models": meta}, indent=2), encoding="utf-8")
    return models


def run(predict, pools, X, keys, clip, p95, base) -> dict[str, object]:
    unclipped = np.asarray(predict(X), dtype=float)
    lookup = dict(zip(keys, np.clip(unclipped, *clip)))
    frames = []
    for seed in SEEDS:
        pool = pools[seed].copy()
        pool["y_pred_saved"] = pool["y_pred"]
        pool["y_pred"] = pool["canonical_key"].map(lookup)
        pool["y_discounted"] = pool["y_pred"] * pool["discount"]
        topk = ssb.select_topk(pool, floor_on=True)
        screened = ssb.screen(topk, radius_threshold=p95)
        screened["seed"] = seed
        screened["p95_final"] = (screened["g1_pass"].astype(bool) & screened["g2_pass"].astype(bool)
                                 & screened["g4_pass"].astype(bool) & screened["radius_mismatch"].le(p95))
        frames.append(screened)
    rows = pd.concat(frames, ignore_index=True)
    records = []
    for key, part in rows.groupby("canonical_key", sort=True):
        n_seeds = part["seed"].nunique()
        passing = part.loc[part["p95_final"], "seed"].nunique()
        records.append({"canonical_key": key, "Formula": part["Formula"].iloc[0], "proto_name": part["proto_name"].iloc[0],
                        "topk_seed_count": n_seeds, "pass_seed_count": passing, "y_pred": float(part["y_pred"].mean()),
                        "consensus": (n_seeds >= 10 and passing == n_seeds and bool(part["g1_pass"].astype(bool).all())
                                      and bool(part["g2_pass"].astype(bool).all())
                                      and bool(part["g4_pass"].astype(bool).all()))})
    union = pd.DataFrame(records).sort_values(["topk_seed_count", "y_pred"], ascending=False).reset_index(drop=True)
    motifs = t.motif_table(rows, base)
    return {"rows": rows, "union": union, "motifs": motifs, "lookup": lookup}


def main() -> None:
    start = time.time()
    evaluator = ssb.CandidateEvaluator()
    p95, _ = t.p95_p90()
    base = {p["name"]: set(p["base_comp"]) for p in ssb.prototypes()}
    pools = {seed: ssb.read_checkpoint_csv(SAVED / "results" / "search_candidates" / f"nsga2_seed{seed}_candidate_pool.csv.gz")
             for seed in SEEDS}
    stored = np.load(FEATURES, allow_pickle=False)
    keys = list(stored["keys"])
    X = pd.DataFrame(stored["values"], columns=list(stored["columns"]))
    X["temperature_K"] = ssb.FIXED_TEMPERATURE_K
    X["log1p_current_rate_C"] = np.log1p(ssb.FIXED_CURRENT_RATE_C)
    X["log1p_cycle_number"] = np.log1p(ssb.FIXED_CYCLE)
    X["log1p_current_rate_x_cycle"] = np.log1p(ssb.FIXED_CURRENT_RATE_C * ssb.FIXED_CYCLE)
    X = X.reindex(columns=evaluator.feature_columns)
    log(f"{len(keys)} pool compositions")

    saved_model = joblib.load(SAVED / "models" / "catboost_full_context.joblib")
    runs = {42: run(lambda Z: saved_model.predict(Z, thread_count=1), pools, X, keys,
                    evaluator.prediction_clip, p95, base)}
    # check against the saved TopK tables
    check = []
    for seed in SEEDS:
        saved = pd.read_csv(SAVED / "results" / "search_candidates" / f"nsga2__floor_on_seed{seed}_topk_screened.csv")
        mine = runs[42]["rows"][runs[42]["rows"]["seed"].eq(seed)]
        check.append({"seed": seed, "same_topk_set": set(saved["canonical_key"]) == set(mine["canonical_key"]),
                      "max_abs_pred_diff_vs_pool": float((mine["y_pred"] - mine["y_pred_saved"]).abs().max())})
    check = pd.DataFrame(check)
    log(f"saved-model check: {check['same_topk_set'].sum()}/20 identical TopK sets")

    for seed, model in train_models().items():
        runs[seed] = run(lambda Z, m=model: m.predict(Z, thread_count=1), pools, X, keys,
                         evaluator.prediction_clip, p95, base)
        log(f"model seed {seed} done")

    from pymatgen.core import Composition
    seven_keys = {f: ssb.canonical_key(Composition(f)) for f in SEVEN + [LI2]}
    summary_rows, seven_rows, motif_rows = [], [], []
    for seed, res in runs.items():
        union, motifs = res["union"], res["motifs"]
        cons = union[union["consensus"]]
        hit = motifs[motifs["motif"].eq(TARGET_MOTIF)]
        summary_rows.append({
            "model_random_seed": seed, "n_consensus": len(cons),
            "seven_recurrent": int(sum(f in set(cons["Formula"]) for f in SEVEN)),
            "consensus_formulas": "; ".join(cons["Formula"]),
            "new_vs_seven": "; ".join(sorted(set(cons["Formula"]) - set(SEVEN))),
            "motif_rank": int(hit["rank"].iloc[0]) if len(hit) else None,
            "motif_position": int(hit["position"].iloc[0]) if len(hit) else None,
            "motif_seed_count": int(hit["seed_count"].iloc[0]) if len(hit) else 0,
            "n_motifs": len(motifs), "n_motifs_20_seeds": int((motifs["seed_count"] == 20).sum()),
            "union_size": len(union)})
        u = union.set_index("canonical_key")
        for f, key in seven_keys.items():
            seven_rows.append({"model_random_seed": seed, "Formula": f,
                               "pred_fixed_context": float(res["lookup"].get(key, np.nan)),
                               "topk_seed_count": int(u.loc[key, "topk_seed_count"]) if key in u.index else 0,
                               "pass_seed_count": int(u.loc[key, "pass_seed_count"]) if key in u.index else 0,
                               "consensus": bool(u.loc[key, "consensus"]) if key in u.index else False})
        top = motifs.head(10).copy()
        top["model_random_seed"] = seed
        motif_rows.append(top)
        union.to_csv(OUT / f"23_recurrence_union_seed{seed}.csv", index=False)
        motifs.to_csv(OUT / f"23_motifs_seed{seed}.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUT / "23_summary_by_model.csv", index=False)
    pd.DataFrame(seven_rows).to_csv(OUT / "23_seven_by_model.csv", index=False)
    pd.concat(motif_rows).to_csv(OUT / "23_top10_motifs_by_model.csv", index=False)
    check.to_csv(OUT / "23_saved_model_check.csv", index=False)
    write_json("23_summary.json", {"p95": p95, "saved_model_check_identical_topk": int(check["same_topk_set"].sum()),
                                   "saved_model_check_max_pred_diff": float(check["max_abs_pred_diff_vs_pool"].max()),
                                   "by_model": summary_rows, "runtime_s": round(time.time() - start)})
    log("done")


if __name__ == "__main__":
    main()
