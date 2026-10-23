"""Candidate and validation checks A-G (analysis 06; see revision_analyses/README.md).

Reads the saved release outputs; B and F also train new CatBoost models with
the original settings.  New models go to models_local/candidate_checks/ and
tables to outputs/06*_*.  No saved model or release file is
overwritten, and no search is run.

  A  anion-contribution fraction = max(0, capacity - strict budget) / capacity
  B  influence of the 12 Cr- or F-containing training compositions (retrain)
  C  cation/anion ratio, two oxidation-state readings, framework site limits
  D  Li2MnO3 template only, NSGA-II vs random search
  E  per-template and per-structure-class G3 thresholds, re-screen
  F  family-holdout leakage check (retrain)
  G  motif-level recurrence under CatBoost and RF ranking

Usage:  python candidate_checks.py [A B C D E F G]   (default: all, in the order A C D E G B F)
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from pymatgen.core import Composition
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1] / "paper_release_v1"
SAVED = HERE.parents[1] / "paper_release_v1"
OUT = HERE.parent / "outputs"
MODEL_OUT = HERE.parent / "models_local" / "candidate_checks"
RF_SEARCH = HERE.parent / "outputs" / "random_forest_reranking" / "search"
sys.path.insert(0, str(WORK / "src"))
sys.path.insert(0, str(HERE))

import search_screen_benchmark as ssb  # noqa: E402
from extended_validation import STRUCTURAL_FAMILY, clustered_metric_ci, point_metrics  # noqa: E402
from rebuild_data_and_models import CB_PARAMS  # noqa: E402
from strict_redox_budget import strict_budget  # noqa: E402

SEEDS = list(range(42, 62))
REF_ARM, RANDOM_ARM = "nsga2__floor_on", "random__floor_on"
SEVEN = [
    "Li2.1Mn0.7Cr0.3O2.865F0.135", "Li2.2Mn0.7Cr0.3O2.865F0.135", "Li2.15Mn0.7Cr0.3O2.865F0.135",
    "Li2.2Mn0.72Cr0.28O2.865F0.135", "Li1.95Mn0.72Cr0.28O2.865F0.135", "Li1.95Mn0.7Cr0.3O2.865F0.135",
    "Li1.9Mn0.7Cr0.3O2.865F0.135",
]
# Framework site limits on (cations incl. P) / (O + F):
#   rocksalt-derived (layered, Li-rich, Li2MnO3): all cations octahedral in a ccp anion lattice -> 1.0
#   spinel LiM2O4: 8a (1) + 16d (2) per 4 O -> 0.75; 16c also filled (Li2M2O4) -> 1.0 (secondary)
#   olivine LiMPO4: M1 (1) + M2 (1) + P tetrahedral (1) per 4 O -> 0.75
SITE_LIMIT = {"layered": 1.0, "spinel": 0.75, "olivine": 0.75}
SITE_LIMIT_SECONDARY = {"spinel": 1.0}
TOL = 1e-9


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def write_json(name: str, data: object) -> None:
    (OUT / name).write_text(json.dumps(data, indent=2, default=float, ensure_ascii=False), encoding="utf-8")


def p95_p90() -> tuple[float, float]:
    known = json.loads((SAVED / "results" / "known_control_summary.json").read_text(encoding="utf-8"))
    return float(known["positive_control_radius_p95_after_g1_g2"]), float(known["positive_control_radius_p90_after_g1_g2"])


def reference_rows(arm: str = REF_ARM, rf: bool = False) -> pd.DataFrame:
    """All TopK rows of one arm over the 20 seeds, with the P95 final-pass flag."""
    p95, _ = p95_p90()
    frames = []
    for seed in SEEDS:
        path = (RF_SEARCH / f"{arm}_seed{seed}_topk_screened_rf.csv" if rf
                else SAVED / "results" / "search_candidates" / f"{arm}_seed{seed}_topk_screened.csv")
        frame = pd.read_csv(path)
        frame["seed"] = seed
        frame["p95_final"] = (frame["g1_pass"].astype(bool) & frame["g2_pass"].astype(bool)
                              & frame["g4_pass"].astype(bool)
                              & pd.to_numeric(frame["radius_mismatch"], errors="coerce").le(p95))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def class_of(proto: str) -> str:
    return STRUCTURAL_FAMILY[proto]


def cation_anion(formula: str) -> float:
    amounts = Composition(formula).get_el_amt_dict()
    anions = sum(x for e, x in amounts.items() if e in ("O", "F"))
    return sum(x for e, x in amounts.items() if e not in ("O", "F")) / anions


def counterparts_table() -> pd.DataFrame:
    return pd.read_csv(OUT / "05_counterpart_controls.csv")


# ----------------------------------------------------------------------------- A
def item_a() -> None:
    log("A: anion-contribution fraction")
    controls = pd.read_csv(OUT / "04_strict_redox_controls.csv")
    for col in ["measured_median", "measured_max", "predicted_y_pred"]:
        cap = controls[col]
        controls[f"anion_fraction_{col}"] = np.maximum(0.0, cap - controls["strict_budget_mAh_g"]) / cap
    controls.to_csv(OUT / "06A_controls_anion_fraction.csv", index=False)

    def dist(part: pd.DataFrame, col: str) -> dict[str, float]:
        v = part[col].to_numpy(float)
        return {"n": int(len(v)), "n_positive": int((v > 0).sum()), "min": float(v.min()),
                "p25": float(np.percentile(v, 25)), "median": float(np.median(v)),
                "p75": float(np.percentile(v, 75)), "max": float(v.max()), "mean": float(v.mean())}

    by_template = {}
    for proto, part in controls.groupby("proto_name", sort=True):
        by_template[proto] = {"median_based": dist(part, "anion_fraction_measured_median"),
                              "max_based": dist(part, "anion_fraction_measured_max")}
    by_class = {}
    controls["class"] = controls["proto_name"].map(class_of)
    for cls, part in controls.groupby("class", sort=True):
        by_class[cls] = {"median_based": dist(part, "anion_fraction_measured_median"),
                         "max_based": dist(part, "anion_fraction_measured_max")}

    table = counterparts_table()
    table["anion_fraction_pred"] = (np.maximum(0.0, table["y_pred_saved_model"] - table["G4_strict_budget"])
                                    / table["y_pred_saved_model"])
    table.to_csv(OUT / "06A_candidates_counterparts_anion_fraction.csv", index=False)

    placement = []
    for proto in ["Li-rich-NMC", "Li2MnO3"]:
        part = controls[controls["proto_name"].eq(proto)]
        for basis in ["measured_median", "measured_max"]:
            ref = part[f"anion_fraction_{basis}"].to_numpy(float)
            for row in table[table["variant"].eq("candidate")].itertuples(index=False):
                value = float(row.anion_fraction_pred)
                placement.append({
                    "control_template": proto, "control_basis": basis, "Formula": row.Formula,
                    "candidate_fraction": value, "n_controls": int(len(ref)),
                    "n_controls_at_or_above": int((ref >= value - TOL).sum()),
                    "percentile_rank": float((ref < value).mean() * 100),
                    "control_max": float(ref.max()), "control_median": float(np.median(ref)),
                })
    placement = pd.DataFrame(placement)
    placement.to_csv(OUT / "06A_candidate_placement.csv", index=False)
    li2mno3_controls = controls[controls["proto_name"].eq("Li2MnO3")][
        ["Formula", "measured_median", "measured_max", "strict_budget_mAh_g",
         "anion_fraction_measured_median", "anion_fraction_measured_max"]]
    write_json("06A_summary.json", {
        "definition": "max(0, capacity - strict budget) / capacity; controls use measured median and max, "
                      "candidates and counterparts use the saved CatBoost prediction (298.15 K, 0.5 C, cycle 1)",
        "controls_by_template": by_template, "controls_by_class": by_class,
        "li2mno3_controls": li2mno3_controls.to_dict(orient="records"),
        "candidates_counterparts": table[["parent_candidate", "variant", "Formula", "y_pred_saved_model",
                                          "G4_strict_budget", "anion_fraction_pred"]].to_dict(orient="records"),
    })


# ----------------------------------------------------------------------------- B
def nominal_features(evaluator: ssb.CandidateEvaluator, formulas: list[str]) -> pd.DataFrame:
    rows = []
    for formula in formulas:
        comp = Composition(formula)
        rows.append(evaluator._composition_row({"comp": comp.get_el_amt_dict(), "Formula": formula,
                                                 "canonical_key": ssb.canonical_key(comp)}))
    X = pd.DataFrame(rows).reindex(columns=evaluator.comp_columns)
    X["temperature_K"] = ssb.FIXED_TEMPERATURE_K
    X["log1p_current_rate_C"] = math.log1p(ssb.FIXED_CURRENT_RATE_C)
    X["log1p_cycle_number"] = math.log1p(ssb.FIXED_CYCLE)
    X["log1p_current_rate_x_cycle"] = math.log1p(ssb.FIXED_CURRENT_RATE_C * ssb.FIXED_CYCLE)
    return X.reindex(columns=evaluator.feature_columns)


def training_table() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    raw = pd.read_csv(SAVED / "data" / "clean_rows.csv").sort_values("row_id").reset_index(drop=True)
    features = pd.read_csv(SAVED / "data" / "features_composition_context.csv")
    if len(raw) != len(features):
        raise RuntimeError("clean_rows and feature table differ in length")
    y = pd.to_numeric(raw["Value"], errors="raise").astype(float)
    return features, y, raw


def item_b() -> None:
    log("B: Cr/F training-composition influence")
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    features, y, raw = training_table()
    evaluator = ssb.CandidateEvaluator()
    if list(features.columns) != list(evaluator.feature_columns):
        raise RuntimeError("Feature order differs from the evaluator")
    assign = pd.read_csv(SAVED / "results" / "known_control_prototype_assignment.csv").set_index("canonical_key")
    groups = raw.groupby("canonical_key").agg(formula=("canonical_formula", "first"), n_records=("Value", "size"),
                                              median=("Value", "median"), min=("Value", "min"),
                                              max=("Value", "max"))
    groups["proto_name"] = assign["proto_name"]
    elements = groups["formula"].map(lambda f: set(Composition(f).get_el_amt_dict()))
    groups["has_Cr"], groups["has_F"] = elements.map(lambda s: "Cr" in s), elements.map(lambda s: "F" in s)
    crf = groups[groups["has_Cr"] | groups["has_F"]].sort_values(["has_F", "formula"]).reset_index()
    crf.to_csv(OUT / "06B_crf_training_compositions.csv", index=False)

    table = counterparts_table()
    formulas = table["Formula"].tolist()
    X = nominal_features(evaluator, formulas)
    saved_raw = np.asarray(evaluator.model.predict(X, thread_count=1), dtype=float)
    clip = evaluator.prediction_clip

    scenarios: list[tuple[str, str, list[str]]] = [("baseline_all_data", "全部資料（檢查用）", [])]
    for row in crf.itertuples(index=False):
        scenarios.append((f"drop_{row.formula}", f"去掉 {row.formula}", [row.canonical_key]))
    scenarios.append(("drop_all_Cr", "去掉 8 個含 Cr 組成", crf.loc[crf["has_Cr"], "canonical_key"].tolist()))
    scenarios.append(("drop_all_F", "去掉 4 個含 F 組成", crf.loc[crf["has_F"], "canonical_key"].tolist()))

    preds = {}
    meta = []
    for name, label, keys in scenarios:
        mask = ~raw["canonical_key"].isin(keys).to_numpy()
        model = CatBoostRegressor(**CB_PARAMS)
        model.fit(features.loc[mask], y.loc[mask])
        model.save_model(str(MODEL_OUT / f"B_{len(meta):02d}_{name.replace('/', '_')}.cbm"))
        preds[name] = np.asarray(model.predict(X, thread_count=1), dtype=float)
        meta.append({"scenario": name, "label": label, "dropped_keys": keys,
                     "n_records_dropped": int((~mask).sum()), "n_records_train": int(mask.sum())})
        log(f"B: {name} done ({int((~mask).sum())} records dropped)")

    base = preds["baseline_all_data"]
    baseline_check = {"max_abs_diff_vs_saved_model": float(np.max(np.abs(base - saved_raw))),
                      "identical": bool(np.array_equal(base, saved_raw)), "prediction_clip": clip}
    long_rows = []
    for name, _, _ in scenarios:
        for i, row in table.iterrows():
            long_rows.append({"scenario": name, "parent_candidate": row["parent_candidate"],
                              "variant": row["variant"], "Formula": row["Formula"],
                              "pred_unclipped": float(preds[name][i]),
                              "pred_clipped": float(np.clip(preds[name][i], *clip)),
                              "baseline_pred_unclipped": float(base[i])})
    long = pd.DataFrame(long_rows)
    long["change_vs_baseline"] = long["pred_unclipped"] - long["baseline_pred_unclipped"]
    long.to_csv(OUT / "06B_predictions_by_scenario.csv", index=False)

    effects = []
    for name, _, _ in scenarios:
        part = long[long["scenario"].eq(name)]
        for parent in SEVEN:
            sub = part[part["parent_candidate"].eq(parent)].set_index("variant")["pred_unclipped"]
            cand, a, b, c = sub["candidate"], sub["(a) no F"], sub["(b) no Cr"], sub["(c) no F, no Cr"]
            base_sub = long[(long["scenario"].eq("baseline_all_data")) & long["parent_candidate"].eq(parent)]
            base_sub = base_sub.set_index("variant")["pred_unclipped"]
            effects.append({
                "scenario": name, "candidate": parent, "candidate_pred": cand,
                "candidate_change_vs_baseline": cand - base_sub["candidate"],
                "Cr_effect_a_minus_c": a - c, "F_effect_b_minus_c": b - c,
                "Cr_effect_change": (a - c) - (base_sub["(a) no F"] - base_sub["(c) no F, no Cr"]),
                "F_effect_change": (b - c) - (base_sub["(b) no Cr"] - base_sub["(c) no F, no Cr"]),
                "above_a": bool(cand > a), "above_b": bool(cand > b), "above_c": bool(cand > c),
                "above_all_three": bool(cand > max(a, b, c)),
            })
    effects = pd.DataFrame(effects)
    effects.to_csv(OUT / "06B_effects_by_scenario.csv", index=False)
    li2mno3 = long[long["variant"].eq("reference")][["scenario", "pred_unclipped", "change_vs_baseline"]]
    write_json("06B_summary.json", {
        "settings": {**CB_PARAMS, "training_table": "saved data/features_composition_context.csv + clean_rows Value",
                     "prediction_context": "298.15 K, 0.5 C, cycle 1 (unclipped model output)"},
        "baseline_check": baseline_check, "scenarios": meta,
        "crf_compositions": crf.to_dict(orient="records"),
        "li2mno3_by_scenario": li2mno3.to_dict(orient="records"),
    })


# ----------------------------------------------------------------------------- C
def readings(formula: str) -> dict[str, float]:
    strict = strict_budget(formula)
    amounts = Composition(formula).get_el_amt_dict()
    mn, cr = amounts.get("Mn", 0.0), amounts.get("Cr", 0.0)
    other_redox = [e for e in amounts if e in ("Ni", "Co", "Fe", "V", "Cu", "Ti")]
    required = strict["tm_required_valence"]
    if other_redox:
        return {"Cr_avg_if_Mn4": np.nan, "Mn_avg_if_Cr3": np.nan}
    return {"Cr_avg_if_Mn4": (required - 4 * mn) / cr if cr else np.nan,
            "Mn_avg_if_Cr3": (required - 3 * cr) / mn if mn else np.nan}


def item_c() -> None:
    log("C: site occupancy and charge compensation")
    table = counterparts_table()
    rows = []
    for row in table.itertuples(index=False):
        rows.append({"group": row.variant, "parent_candidate": row.parent_candidate, "Formula": row.Formula,
                     "cation_to_anion": cation_anion(row.Formula), **readings(row.Formula)})
    controls = pd.read_csv(SAVED / "results" / "known_control_screened.csv")
    for row in controls[controls["proto_name"].isin(["Li2MnO3", "Li-rich-NMC"])].itertuples(index=False):
        rows.append({"group": f"control {row.proto_name}", "parent_candidate": "", "Formula": row.Formula,
                     "cation_to_anion": cation_anion(row.Formula), **readings(row.Formula)})
    ratio_table = pd.DataFrame(rows)
    ratio_table.to_csv(OUT / "06C_ratio_and_valence_readings.csv", index=False)

    def above(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        frame["class"] = frame["proto_name"].map(class_of)
        frame["cation_to_anion"] = frame["Formula"].map(cation_anion)
        frame["limit"] = frame["class"].map(SITE_LIMIT)
        frame["above_limit"] = frame["cation_to_anion"] > frame["limit"] + TOL
        frame["limit_secondary"] = frame["class"].map(SITE_LIMIT_SECONDARY)
        frame["above_secondary"] = frame["cation_to_anion"] > frame["limit_secondary"] + TOL
        return frame

    survivors = above(reference_rows())
    survivors = survivors[survivors["p95_final"]]
    by_template = survivors.groupby("proto_name", sort=True).agg(
        cls=("class", "first"), limit=("limit", "first"), n_rows=("Formula", "size"),
        n_rows_above=("above_limit", "sum"), n_unique=("canonical_key", "nunique"),
        ratio_min=("cation_to_anion", "min"), ratio_median=("cation_to_anion", "median"),
        ratio_max=("cation_to_anion", "max")).reset_index()
    uniq = survivors.drop_duplicates("canonical_key")
    by_template = by_template.merge(uniq.groupby("proto_name")["above_limit"].sum().rename("n_unique_above"),
                                    on="proto_name")
    secondary = survivors[survivors["class"].eq("spinel")].groupby("proto_name")["above_secondary"].sum()
    by_template["n_rows_above_secondary"] = by_template["proto_name"].map(secondary)
    by_template["frac_rows_above"] = by_template["n_rows_above"] / by_template["n_rows"]
    by_template["frac_unique_above"] = by_template["n_unique_above"] / by_template["n_unique"]
    by_template.to_csv(OUT / "06C_survivors_site_limit_by_template.csv", index=False)

    ctrl = above(controls)
    ctrl_by = ctrl.groupby("proto_name", sort=True).agg(
        n=("Formula", "size"), n_above=("above_limit", "sum"), ratio_min=("cation_to_anion", "min"),
        ratio_max=("cation_to_anion", "max")).reset_index()
    ctrl_by.to_csv(OUT / "06C_controls_site_limit_by_template.csv", index=False)
    write_json("06C_summary.json", {
        "site_limits": SITE_LIMIT, "site_limits_secondary": SITE_LIMIT_SECONDARY,
        "survivor_rows_total": int(len(survivors)), "survivor_rows_above": int(survivors["above_limit"].sum()),
        "survivor_unique_total": int(uniq["canonical_key"].nunique()),
        "survivor_unique_above": int(uniq["above_limit"].sum()),
        "by_template": by_template.to_dict(orient="records"), "controls": ctrl_by.to_dict(orient="records"),
        "ratios": ratio_table.to_dict(orient="records"),
    })


# ----------------------------------------------------------------------------- D
def item_d() -> None:
    log("D: template-only search Li2MnO3 NSGA-II vs random")
    seven_keys = {ssb.canonical_key(Composition(f)): f for f in SEVEN}
    per_seed = []
    survivor_sets = {}
    for arm in [REF_ARM, RANDOM_ARM]:
        rows = reference_rows(arm)
        topk = rows[rows["proto_name"].eq("Li2MnO3")]
        surv = topk[topk["p95_final"]]
        survivor_sets[arm] = surv
        for seed in SEEDS:
            s = surv[surv["seed"].eq(seed)].sort_values("y_pred", ascending=False)
            t = topk[topk["seed"].eq(seed)]
            per_seed.append({
                "arm": arm, "seed": seed, "n_topk_li2mno3": int(len(t)), "n_p95_survivors": int(len(s)),
                "max_y_pred": float(s["y_pred"].max()) if len(s) else np.nan,
                "mean_top10_y_pred": float(s["y_pred"].head(10).mean()) if len(s) else np.nan,
                "n_in_top10": int(min(10, len(s))),
                "median_comp_novelty": float(s["comp_novelty"].median()) if len(s) else np.nan,
                "mean_top10_comp_novelty": float(s["comp_novelty"].head(10).mean()) if len(s) else np.nan,
                "n_seven_in_topk": int(t["canonical_key"].isin(seven_keys).sum()),
                "n_seven_in_survivors": int(s["canonical_key"].isin(seven_keys).sum()),
            })
    per_seed = pd.DataFrame(per_seed)
    per_seed.to_csv(OUT / "06D_per_seed.csv", index=False)

    paired = {}
    wide = per_seed.pivot(index="seed", columns="arm")
    for metric in ["n_p95_survivors", "max_y_pred", "mean_top10_y_pred", "median_comp_novelty",
                   "mean_top10_comp_novelty"]:
        a, b = wide[(metric, REF_ARM)], wide[(metric, RANDOM_ARM)]
        ok = a.notna() & b.notna()
        diff = (a - b)[ok]
        test = wilcoxon(a[ok], b[ok]) if ok.sum() >= 5 and (diff != 0).any() else None
        paired[metric] = {"nsga2_mean": float(a.mean()), "random_mean": float(b.mean()),
                          "n_seeds_both": int(ok.sum()), "mean_diff": float(diff.mean()),
                          "seeds_nsga2_higher": int((diff > 0).sum()), "seeds_random_higher": int((diff < 0).sum()),
                          "seeds_equal": int((diff == 0).sum()),
                          "wilcoxon_p": float(test.pvalue) if test is not None else None}

    random_counts = (survivor_sets[RANDOM_ARM].groupby(["canonical_key", "Formula"])["seed"].nunique()
                     .rename("seed_count").reset_index().sort_values("seed_count", ascending=False))
    random_counts[random_counts["seed_count"] >= 10].to_csv(OUT / "06D_random_recurrent_li2mno3.csv", index=False)
    recurrence = {}
    for arm, surv in survivor_sets.items():
        counts = surv.groupby("canonical_key")["seed"].nunique()
        recurrence[arm] = {"unique_survivors": int(len(counts)), "in_10plus_seeds": int((counts >= 10).sum()),
                           "max_seed_count": int(counts.max()) if len(counts) else 0,
                           "seven_seed_counts": {f: int(counts.get(k, 0)) for k, f in seven_keys.items()}}
    # The full random candidate pools (before TopK selection).
    pool_hits = {f: 0 for f in SEVEN}
    pool_hits_nsga = {f: 0 for f in SEVEN}
    for proposal, hits in [("random", pool_hits), ("nsga2", pool_hits_nsga)]:
        for seed in SEEDS:
            pool = pd.read_csv(SAVED / "results" / "search_candidates" / f"{proposal}_seed{seed}_candidate_pool.csv.gz",
                               usecols=["canonical_key"])
            present = set(pool["canonical_key"])
            for key, f in seven_keys.items():
                hits[f] += int(key in present)
    write_json("06D_summary.json", {"paired": paired, "recurrence": recurrence,
                                   "seven_in_random_pool_seed_count": pool_hits,
                                   "seven_in_nsga2_pool_seed_count": pool_hits_nsga})


# ----------------------------------------------------------------------------- E
def item_e() -> None:
    log("E: class G3 thresholds per-template and per-class G3 thresholds")
    p95, p90 = p95_p90()
    controls = pd.read_csv(SAVED / "results" / "known_control_screened.csv")
    reached = controls[controls["pass_g1_g2"].astype(bool)].copy()
    reached["class"] = reached["proto_name"].map(class_of)
    rows = []
    for level, key in [("template", "proto_name"), ("class", "class")]:
        for name, part in reached.groupby(key, sort=True):
            v = part["radius_mismatch"].to_numpy(float)
            rows.append({"level": level, "name": name, "n_reached_g3": int(len(v)),
                         "n_controls": int(controls[controls["proto_name"].map(class_of if level == "class" else
                                                                               (lambda x: x)).eq(name)].shape[0]),
                         "P90": float(np.quantile(v, 0.90)) if len(v) >= 10 else np.nan,
                         "P95": float(np.quantile(v, 0.95)) if len(v) >= 10 else np.nan,
                         "min": float(v.min()), "median": float(np.median(v)), "max": float(v.max())})
    v_all = reached["radius_mismatch"].to_numpy(float)
    rows.append({"level": "pooled", "name": "all", "n_reached_g3": int(len(v_all)), "n_controls": int(len(controls)),
                 "P90": p90, "P95": p95, "min": float(v_all.min()), "median": float(np.median(v_all)),
                 "max": float(v_all.max())})
    thresholds = pd.DataFrame(rows)
    thresholds.to_csv(OUT / "06E_thresholds.csv", index=False)
    cls_t = thresholds[thresholds["level"].eq("class")].set_index("name")

    rows_all = reference_rows()
    rows_all["class"] = rows_all["proto_name"].map(class_of)
    radius = pd.to_numeric(rows_all["radius_mismatch"], errors="coerce")
    g124 = rows_all["g1_pass"].astype(bool) & rows_all["g2_pass"].astype(bool) & rows_all["g4_pass"].astype(bool)
    original = pd.read_csv(SAVED / "results" / "consensus_candidates.csv")
    results = {}
    for q in ["P95", "P90"]:
        thr = rows_all["class"].map(cls_t[q])
        rows_all[f"class_{q}_final"] = g124 & radius.le(thr)
        records = []
        for key, part in rows_all.groupby("canonical_key", sort=True):
            seeds = part["seed"].nunique()
            passing = part.loc[part[f"class_{q}_final"], "seed"].nunique()
            records.append({"canonical_key": key, "Formula": part["Formula"].iloc[0],
                            "proto_name": part["proto_name"].iloc[0], "topk_seed_count": seeds,
                            "pass_seed_count": passing, "radius_mismatch": float(radius[part.index].median()),
                            "consensus": seeds >= 10 and passing == seeds and bool(g124[part.index].all())})
        union = pd.DataFrame(records)
        union.to_csv(OUT / f"06E_class_{q}_recurrence_union.csv", index=False)
        cons = union[union["consensus"]].sort_values("topk_seed_count", ascending=False)
        per_seed = rows_all.groupby("seed")[f"class_{q}_final"].sum()
        results[q] = {
            "class_thresholds": cls_t[q].to_dict(), "union": int(len(union)), "consensus_n": int(len(cons)),
            "consensus": cons[["Formula", "proto_name", "topk_seed_count", "radius_mismatch"]].to_dict(orient="records"),
            "original_retained": sorted(set(original["Formula"]) & set(cons["Formula"])),
            "original_dropped": sorted(set(original["Formula"]) - set(cons["Formula"])),
            "new": sorted(set(cons["Formula"]) - set(original["Formula"])),
            "per_seed_survivors_mean": float(per_seed.mean()),
            "per_seed_survivors_min": int(per_seed.min()), "per_seed_survivors_max": int(per_seed.max()),
        }
    layered = reached[reached["class"].eq("layered")]["radius_mismatch"].to_numpy(float)
    seven_g3 = []
    for f in SEVEN:
        rms, _ = ssb.radius_mismatch_metrics(f, "Li2MnO3")
        seven_g3.append({"Formula": f, "G3": float(rms), "n_layered_controls": int(len(layered)),
                         "n_layered_controls_below": int((layered < rms).sum()),
                         "percentile_in_layered": float((layered < rms).mean() * 100),
                         "layered_P95": float(cls_t.loc["layered", "P95"]),
                         "layered_P90": float(cls_t.loc["layered", "P90"]),
                         "passes_layered_P95": bool(rms <= cls_t.loc["layered", "P95"]),
                         "passes_layered_P90": bool(rms <= cls_t.loc["layered", "P90"])})
    pd.DataFrame(seven_g3).to_csv(OUT / "06E_seven_g3_in_layered.csv", index=False)
    write_json("06E_summary.json", {"pooled_p95_saved": p95, "pooled_p90_saved": p90,
                                   "thresholds": thresholds.to_dict(orient="records"),
                                   "rescreen": results, "seven_g3": seven_g3,
                                   "layered_controls_g3_sorted": sorted(map(float, layered))})


# ----------------------------------------------------------------------------- F
def frac_vector(key: str) -> np.ndarray:
    parts = dict(item.split(":") for item in key.split("|"))
    vec = np.zeros(len(ssb.PERIODIC_ELEMENTS))
    for i, e in enumerate(ssb.PERIODIC_ELEMENTS):
        if e in parts:
            vec[i] = float(parts[e])
    return vec / vec.sum()


def family_holdout(features: pd.DataFrame, y: pd.Series, raw: pd.DataFrame, family_by_key: pd.Series,
                   extra_drop: dict[str, list[str]], tag: str) -> tuple[pd.DataFrame, dict[str, object]]:
    row_family = raw["canonical_key"].map(family_by_key)
    params = dict(CB_PARAMS)
    params["thread_count"] = 1
    predictions = []
    dropped = {}
    for family_index, family in enumerate(["layered", "olivine", "spinel"]):
        test_mask = row_family.eq(family).to_numpy()
        extra = raw["canonical_key"].isin(extra_drop.get(family, [])).to_numpy()
        train_mask = ~test_mask & ~extra
        dropped[family] = {"extra_compositions": len(extra_drop.get(family, [])), "extra_records": int(extra.sum()),
                           "train_records": int(train_mask.sum())}
        params["random_seed"] = 43000 + family_index
        model = CatBoostRegressor(**params)
        model.fit(features.loc[train_mask], y.loc[train_mask])
        model.save_model(str(MODEL_OUT / f"F_{tag}_{family}.cbm"))
        predicted = np.asarray(model.predict(features.loc[test_mask], thread_count=1), dtype=float)
        predictions.append(pd.DataFrame({
            "row_id": raw.loc[test_mask, "row_id"].to_numpy(int),
            "canonical_key": raw.loc[test_mask, "canonical_key"].astype(str).to_numpy(),
            "held_out_family": family, "y_true": y.loc[test_mask].to_numpy(float), "y_pred": predicted}))
        log(f"F: {tag} {family} done")
    combined = pd.concat(predictions, ignore_index=True).sort_values("row_id").reset_index(drop=True)
    combined.to_csv(OUT / f"06F_{tag}_predictions.csv", index=False)
    summary = {"overall": point_metrics(combined["y_true"].to_numpy(float), combined["y_pred"].to_numpy(float)),
               "overall_ci95": clustered_metric_ci(combined, seed=42019),
               "n_records": int(len(combined)), "by_family": {}, "training": dropped}
    for family, part in combined.groupby("held_out_family", sort=True):
        summary["by_family"][family] = {"n_records": int(len(part)),
                                        **point_metrics(part["y_true"].to_numpy(float), part["y_pred"].to_numpy(float))}
    return combined, summary


def item_f() -> None:
    log("F: family-holdout family-holdout leakage check")
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    features, y, raw = training_table()
    assign = pd.read_csv(SAVED / "results" / "known_control_prototype_assignment.csv")
    templated = assign.dropna(subset=["proto_name"]).set_index("canonical_key")["proto_name"].astype(str)
    family_by_key = templated.map(STRUCTURAL_FAMILY)
    untemplated = assign[assign["proto_name"].isna()][["canonical_key", "canonical_formula", "n_records"]]
    keys = list(family_by_key.index) + list(untemplated["canonical_key"])
    vec = {k: frac_vector(k) for k in keys}

    flag_rows = []
    extra_drop: dict[str, list[str]] = {}
    for family in ["layered", "olivine", "spinel"]:
        members = list(family_by_key[family_by_key.eq(family)].index)
        M = np.array([vec[k] for k in members])
        within = np.abs(M[:, None, :] - M[None, :, :]).sum(axis=2)
        np.fill_diagonal(within, np.inf)
        nn_within = within.min(axis=1)
        for u in untemplated.itertuples(index=False):
            d = np.abs(M - vec[u.canonical_key]).sum(axis=1)
            closer = d < nn_within
            j = int(np.argmin(d))
            flag_rows.append({
                "family": family, "untemplated_formula": u.canonical_formula, "canonical_key": u.canonical_key,
                "n_records": int(u.n_records), "nearest_member": assign.set_index("canonical_key").loc[members[j], "canonical_formula"],
                "nearest_member_proto": templated[members[j]], "distance_to_nearest_member": float(d[j]),
                "that_member_within_family_nn": float(nn_within[j]),
                "n_members_closer_than_their_nn": int(closer.sum()),
                "flagged": bool(closer.any()),
                "members_closer": ";".join(assign.set_index("canonical_key").loc[members[i], "canonical_formula"]
                                           for i in np.where(closer)[0]),
            })
            if closer.any():
                extra_drop.setdefault(family, []).append(u.canonical_key)
    flags = pd.DataFrame(flag_rows)
    flags.to_csv(OUT / "06F_untemplated_distances.csv", index=False)
    log(f"F: flagged per family {[(f, len(v)) for f, v in extra_drop.items()]}")

    _, original = family_holdout(features, y, raw, family_by_key, {}, "replicate_original")
    saved = json.loads((SAVED / "results" / "leave_one_family_out_validation.json").read_text(encoding="utf-8"))
    _, cleaned = family_holdout(features, y, raw, family_by_key, extra_drop, "flagged_removed")
    write_json("06F_summary.json", {
        "flag_rule": "untemplated composition u is flagged for family f if, for some member m of f, "
                     "Manhattan(u, m) < Manhattan(m, nearest other member of f); atomic-fraction vectors",
        "flagged_by_family": {f: flags[(flags["family"].eq(f)) & flags["flagged"]]["untemplated_formula"].tolist()
                              for f in ["layered", "olivine", "spinel"]},
        "saved_overall": saved["overall"], "replicate_original": original, "flagged_removed": cleaned,
    })


# ----------------------------------------------------------------------------- G
def motif_table(rows: pd.DataFrame, base: dict[str, set[str]]) -> pd.DataFrame:
    surv = rows[rows["p95_final"]].copy()
    surv["substituents"] = [
        "+".join(sorted(set(Composition(f).get_el_amt_dict()) - base[p])) or "(none)"
        for f, p in zip(surv["Formula"], surv["proto_name"])]
    surv["motif"] = surv["proto_name"] + " | " + surv["substituents"]
    per_seed = surv.groupby(["motif", "seed"])["canonical_key"].nunique().rename("n_formulas").reset_index()
    out = per_seed.groupby("motif").agg(
        seed_count=("seed", "nunique"), formulas_per_seed_mean=("n_formulas", "mean"),
        formulas_per_seed_min=("n_formulas", "min"), formulas_per_seed_max=("n_formulas", "max")).reset_index()
    total = surv.groupby("motif")["canonical_key"].nunique().rename("distinct_formulas_all_seeds")
    out = out.merge(total, on="motif")
    out["seeds_list"] = out["motif"].map(per_seed.groupby("motif")["seed"].apply(lambda s: ";".join(map(str, sorted(s)))))
    out = out.sort_values(["seed_count", "formulas_per_seed_mean"], ascending=[False, False]).reset_index(drop=True)
    out["rank"] = out["seed_count"].rank(method="min", ascending=False).astype(int)
    out["position"] = np.arange(1, len(out) + 1)
    return out


def item_g() -> None:
    log("G: motif-level recurrence")
    base = {p["name"]: set(p["base_comp"]) for p in ssb.prototypes()}
    target = "Li2MnO3 | Cr+F"
    summary = {"motif_definition": "template + set of elements not in the template base composition (ratios ignored)",
               "survivor_rule": "P95 final (G1, G2, G4 and G3 <= P95) in the reference arm nsga2__floor_on",
               "rank_rule": "rank = number of seeds (ties share the minimum rank); listed order breaks ties by "
                            "mean distinct formulas per seed"}
    for label, rf in [("catboost", False), ("rf", True)]:
        table = motif_table(reference_rows(REF_ARM, rf=rf), base)
        table.to_csv(OUT / f"06G_motifs_{label}.csv", index=False)
        hit = table[table["motif"].eq(target)]
        summary[label] = {"n_motifs": int(len(table)), "top10": table.head(10).to_dict(orient="records"),
                          "target": hit.to_dict(orient="records")[0] if len(hit) else None,
                          "n_motifs_10plus_seeds": int((table["seed_count"] >= 10).sum()),
                          "n_motifs_20_seeds": int((table["seed_count"] == 20).sum())}
    write_json("06G_summary.json", summary)


ITEMS = {"A": item_a, "B": item_b, "C": item_c, "D": item_d, "E": item_e, "F": item_f, "G": item_g}

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    chosen = [a.upper() for a in sys.argv[1:]] or ["A", "C", "D", "E", "G", "B", "F"]
    for item in chosen:
        start = time.time()
        ITEMS[item]()
        log(f"{item} finished in {time.time() - start:.0f} s")
