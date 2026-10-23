"""Li = 2 member of the Cr/F-Li2MnO3 motif and motif-level check of the 14 sensitivity models (analysis 17; see revision_analyses/README.md).

1a  Li2Mn0.7Cr0.3O2.865F0.135: saved-model prediction (via the original CandidateEvaluator), 100 saved
    bootstrap models P05-P95, 16-context minimum, G1-G4 and P95, cation/anion ratio, strict cationic budget,
    minimum non-cationic share = max(0, prediction - strict budget) / prediction, two oxidation-state readings
1b  paired bootstrap differences against its counterparts (a) no F, (b) no Cr, (c) no F and no Cr (= Li2MnO3)
1c  occurrences in the NSGA-II and random candidate pools, TopK tables and P95 survivors over the 20 seeds,
    and the step at which it leaves the NSGA-II recurrence rule
2   the two 14 retrained models (exclude V_high >= 4.8 V; exclude V_high < 4.6 V): the 7 candidates, the
    Li = 2 member and their (a), (b), (c) counterparts; candidate - counterpart differences vs the saved model
Saved artifacts and the two 14 models only; nothing is trained and no search is run.  Outputs: outputs/17_*.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from pymatgen.core import Composition

import candidate_checks as t
from composition_counterparts import TABLE3, counterparts, load_models
from literature_pairs import features
from strict_redox_budget import strict_budget
from candidate_checks import OUT, SAVED, SEEDS, log, ssb, write_json

LI2 = "Li2Mn0.7Cr0.3O2.865F0.135"
PROTO = "Li2MnO3"
CONTEXTS = [(r, c) for r in [0.1, 0.5, 1.0, 5.0] for c in [1.0, 50.0, 100.0, 200.0]]
SENS_MODELS = {"exclude_Vhigh_ge_4.8": "upper_voltage_sensitivity_models/exclude_Vhigh_ge_4.8.cbm",
               "exclude_Vhigh_lt_4.6": "upper_voltage_sensitivity_models/exclude_Vhigh_lt_4.6.cbm"}


def norm(f: str) -> str:
    return Composition(f).formula.replace(" ", "")


def item_1a_1b(evaluator, models) -> dict:
    p95 = t.p95_p90()[0]
    comp = Composition(LI2)
    ev = evaluator.evaluate([{"comp": comp.get_el_amt_dict(), "Formula": norm(LI2),
                              "canonical_key": ssb.canonical_key(comp), "proto_name": PROTO, "is_doped": 1}])
    y_pred = float(ev["y_pred"].iloc[0])
    X = features(evaluator, [LI2], 0.5, 1.0)
    boot = np.array([float(m.predict(X, thread_count=1)[0]) for m in models])
    ctx = [float(evaluator.model.predict(features(evaluator, [LI2], r, c), thread_count=1)[0]) for r, c in CONTEXTS]
    g1 = ssb.gate1_ionicity(LI2)
    g2 = ssb.gate2_li_tm(LI2, PROTO)
    rms, rmax = ssb.radius_mismatch_metrics(LI2, PROTO)
    _, q_cat, n_e = ssb.redox_budget(LI2)
    strict = strict_budget(LI2)
    readings = t.readings(LI2) if hasattr(t, "readings") else {}
    a = {
        "Formula": LI2, "y_pred_saved_model": y_pred, "y_pred_unclipped": float(ev["y_pred_unclipped"].iloc[0]),
        "bootstrap_P05": float(np.percentile(boot, 5)), "bootstrap_P95": float(np.percentile(boot, 95)),
        "bootstrap_median": float(np.median(boot)), "context_min_16": float(min(ctx)),
        "context_argmin": CONTEXTS[int(np.argmin(ctx))], "context_all": dict(zip(map(str, CONTEXTS), ctx)),
        "G1_pass": bool(g1[0]), "G1_min_abs_charge": float(g1[1]),
        "G2_pass": bool(g2[0]), "Li_TM_ratio": float(g2[1]),
        "G3_weighted_rms": float(rms), "G3_pass_P95": bool(rms <= p95), "G3_pass_P90": bool(rms <= t.p95_p90()[1]),
        "G4_budget": float(q_cat), "G4_redox_electrons": float(n_e), "G4_pass": bool(y_pred <= q_cat),
        "P95_final": bool(g1[0] and g2[0] and rms <= p95 and y_pred <= q_cat),
        "cation_to_anion": strict["cation_to_anion_ratio"], "strict_budget": strict["strict_budget_mAh_g"],
        "strict_G4_pass": bool(y_pred <= strict["strict_budget_mAh_g"]),
        "min_noncationic_share": max(0.0, y_pred - strict["strict_budget_mAh_g"]) / y_pred,
        "model_distance": float(ev["model_distance"].iloc[0]), "comp_novelty": float(ev["comp_novelty"].iloc[0]),
        **readings,
    }
    # 1b paired comparisons
    others = counterparts(LI2)
    formulas = [norm(LI2)] + [norm(f) for _, f in others]
    Xp = features(evaluator, formulas, 0.5, 1.0)
    bp = np.column_stack([np.asarray(m.predict(Xp, thread_count=1), dtype=float) for m in models])
    full = np.asarray(evaluator.model.predict(Xp, thread_count=1), dtype=float)
    pairs = []
    for j, (label, _) in enumerate(others, start=1):
        d = bp[0] - bp[j]
        pairs.append({"comparison": label, "counterpart": formulas[j], "full_model_diff": float(full[0] - full[j]),
                      "median": float(np.median(d)), "P05": float(np.percentile(d, 5)), "P95": float(np.percentile(d, 95)),
                      "n_positive": int((d > 0).sum())})
    return {"a": a, "b": pairs}


def item_1c() -> dict:
    p95 = t.p95_p90()[0]
    key = ssb.canonical_key(Composition(LI2))
    out = {"canonical_key": key}
    for proposal in ["nsga2", "random"]:
        arm = f"{proposal}__floor_on"
        pool_seeds, topk_seeds, surv_seeds, gate_fail = [], [], [], {}
        for seed in SEEDS:
            pool = pd.read_csv(SAVED / "results" / "search_candidates" / f"{proposal}_seed{seed}_candidate_pool.csv.gz",
                               usecols=["canonical_key"])
            if key in set(pool["canonical_key"]):
                pool_seeds.append(seed)
            topk = pd.read_csv(SAVED / "results" / "search_candidates" / f"{arm}_seed{seed}_topk_screened.csv")
            hit = topk[topk["canonical_key"].eq(key)]
            if len(hit):
                topk_seeds.append(seed)
                r = hit.iloc[0]
                ok = (bool(r["g1_pass"]) and bool(r["g2_pass"]) and bool(r["g4_pass"])
                      and float(r["radius_mismatch"]) <= p95)
                if ok:
                    surv_seeds.append(seed)
                else:
                    gate_fail[seed] = {g: bool(r[g]) for g in ["g1_pass", "g2_pass", "g4_pass"]} | {
                        "radius_le_p95": float(r["radius_mismatch"]) <= p95}
        out[proposal] = {"pool_seeds": pool_seeds, "topk_seeds": topk_seeds, "p95_survivor_seeds": surv_seeds,
                         "n_pool": len(pool_seeds), "n_topk": len(topk_seeds), "n_survivor": len(surv_seeds),
                         "gate_failures_in_topk": gate_fail}
    # NSGA-II pool rows: model prediction and selection score for this composition (first seed where it appears)
    rows = []
    for seed in out["nsga2"]["pool_seeds"]:
        pool = ssb.read_checkpoint_csv(SAVED / "results" / "search_candidates" / f"nsga2_seed{seed}_candidate_pool.csv.gz")
        part = pool[pool["canonical_key"].eq(key)]
        li2mno3 = pool[pool["proto_name"].eq(PROTO)]
        topk = pd.read_csv(SAVED / "results" / "search_candidates" / f"nsga2__floor_on_seed{seed}_topk_screened.csv")
        tk = topk[topk["proto_name"].eq(PROTO)]
        rows.append({"seed": seed, "n_rows_in_pool": int(len(part)),
                     "y_pred": float(part["y_pred"].iloc[0]) if "y_pred" in part else math.nan,
                     "y_discounted": float(part["y_discounted"].iloc[0]) if "y_discounted" in part else math.nan,
                     "comp_novelty": float(part["comp_novelty"].iloc[0]) if "comp_novelty" in part else math.nan,
                     "pareto_rank": float(part["pareto_rank"].min()) if "pareto_rank" in part else math.nan,
                     "li2mno3_rows_in_pool": int(len(li2mno3)), "li2mno3_rows_in_topk": int(len(tk)),
                     "topk_min_y_discounted_li2mno3": float(tk["y_discounted"].min()) if len(tk) else math.nan,
                     "in_topk": seed in out["nsga2"]["topk_seeds"]})
    # Re-rank each NSGA-II pool exactly as select_topk does (Pareto rank on y_discounted and comp_novelty,
    # then crowding distance) to see where the Li = 2 member falls relative to the TopK cut.
    seven_keys = {ssb.canonical_key(Composition(f)): f for f, _, _ in TABLE3}
    for row in rows:
        seed = row["seed"]
        pool = ssb.read_checkpoint_csv(SAVED / "results" / "search_candidates" / f"nsga2_seed{seed}_candidate_pool.csv.gz")
        ranked = ssb.rank_pool(pool)
        dedup = ranked.drop_duplicates("canonical_key").reset_index(drop=True)
        me = dedup[dedup["canonical_key"].eq(key)].iloc[0]
        topk = pd.read_csv(SAVED / "results" / "search_candidates" / f"nsga2__floor_on_seed{seed}_topk_screened.csv")
        in_topk = dedup[dedup["canonical_key"].isin(set(topk["canonical_key"]))]
        f1, f2 = float(me["y_discounted"]), float(me["comp_novelty"])
        dom = dedup[(dedup["y_discounted"] >= f1) & (dedup["comp_novelty"] >= f2)
                    & ((dedup["y_discounted"] > f1) | (dedup["comp_novelty"] > f2))]
        # replicate select_topk(floor_on=True): per-template floor first, then fill by overall order
        used: set[str] = set()
        floor_keys = set()
        for proto_name in sorted(ranked["proto_name"].unique()):
            part = ssb.unique_in_order(ranked[ranked["proto_name"] == proto_name], ssb.FLOOR_PER_PROTO, used)
            floor_keys |= set(part["canonical_key"])
        remaining = ssb.unique_in_order(ranked, ssb.TOPK - len(floor_keys), used)
        fill_keys = list(remaining["canonical_key"])
        pos = {k: i + 1 for i, k in enumerate(dedup["canonical_key"])}
        replicated = floor_keys | set(fill_keys)
        row.update({"replicated_topk_equals_saved": replicated == set(topk["canonical_key"]),
                    "n_floor": len(floor_keys), "n_fill": len(fill_keys),
                    "fill_cut_position_unique": pos[fill_keys[-1]],
                    "fill_cut_pareto_rank": int(dedup.loc[pos[fill_keys[-1]] - 1, "pareto_rank"]),
                    "pool_pareto_rank": int(me["pareto_rank"]),
                    "pool_position_unique": int(dedup.index[dedup["canonical_key"].eq(key)][0]) + 1,
                    "topk_max_pool_rank": int(in_topk["pareto_rank"].max()),
                    "topk_last_position_unique": int(in_topk.index.max()) + 1,
                    "n_dominating": int(len(dom)),
                    "dominating_seven": ";".join(sorted(seven_keys[k] for k in dom["canonical_key"] if k in seven_keys)),
                    "n_dominating_li2mno3": int(dom["proto_name"].eq(PROTO).sum())})
    pd.DataFrame(rows).to_csv(OUT / "17_li2_nsga2_pool_rows.csv", index=False, encoding="utf-8-sig")
    out["nsga2_pool_rows"] = rows
    union = pd.read_csv(SAVED / "results" / "candidate_recurrence_union.csv")
    hit = union[union["canonical_key"].eq(key)]
    out["in_saved_recurrence_union"] = hit.to_dict(orient="records")
    return out


def item_2(evaluator) -> pd.DataFrame:
    members = [f for f, _, _ in TABLE3] + [LI2]
    entries = []
    for f in members:
        entries.append((f, "candidate", norm(f)))
        entries += [(f, label, norm(o)) for label, o in counterparts(f)]
    X = features(evaluator, [e[2] for e in entries], 0.5, 1.0)
    preds = {"saved": np.asarray(evaluator.model.predict(X, thread_count=1), dtype=float)}
    for name, path in SENS_MODELS.items():
        m = CatBoostRegressor()
        m.load_model(str(t.HERE.parent / "models_local" / path))
        preds[name] = np.asarray(m.predict(X, thread_count=1), dtype=float)
    rows = []
    for i, (parent, label, f) in enumerate(entries):
        rows.append({"member": parent, "variant": label, "Formula": f, **{k: float(v[i]) for k, v in preds.items()}})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "17_sensitivity_members_predictions.csv", index=False, encoding="utf-8-sig")
    diffs = []
    for parent, part in table.groupby("member", sort=False):
        cand = part[part["variant"].eq("candidate")].iloc[0]
        for _, r in part[~part["variant"].eq("candidate")].iterrows():
            d = {"member": parent, "comparison": r["variant"], "counterpart": r["Formula"]}
            for k in preds:
                d[f"diff_{k}"] = float(cand[k] - r[k])
                d[f"above_{k}"] = bool(cand[k] > r[k])
            diffs.append(d)
    out = pd.DataFrame(diffs)
    for k in SENS_MODELS:
        out[f"diff_change_{k}"] = out[f"diff_{k}"] - out["diff_saved"]
    out.to_csv(OUT / "17_sensitivity_members_diffs.csv", index=False, encoding="utf-8-sig")
    return out


def main() -> None:
    evaluator = ssb.CandidateEvaluator()
    models, cols = load_models()
    if cols != list(evaluator.feature_columns):
        raise RuntimeError("Bootstrap model feature order differs from the evaluator")
    log("1a/1b")
    res = item_1a_1b(evaluator, models)
    log("1c")
    res["c"] = item_1c()
    log("2")
    item_2(evaluator)
    write_json("17_summary.json", res)


if __name__ == "__main__":
    main()
