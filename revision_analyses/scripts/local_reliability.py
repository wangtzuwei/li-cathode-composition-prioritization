"""20: local reliability near the candidates (analysis 20; see revision_analyses/README.md).

a  Saved composition-holdout (LOCO, one canonical composition per fold) out-of-fold predictions of the CatBoost
   composition + context model (results/oof_canonical_composition_heldout_all_models.csv, model
   catboost__composition_plus_context): MAE, RMSE, R2, Spearman on all 907 records and on subsets
   (Li2MnO3 template, Li-rich-NMC template, Cr-containing, F-containing), with the composition-clustered
   bootstrap 95% CI of the release (extended_validation.clustered_metric_ci).  Template = the release's
   prototype assignment of the 171 training compositions (known_control_prototype_assignment.csv).
b  5NN distance of the 7 candidates and the Li = 2 member in the search evaluator's scaled composition space
   (mean distance to the 5 nearest training compositions), its percentile among the 171 training compositions'
   own 5NN distances (self excluded, as the evaluator computes model_distance_p95), and whether it is below the
   P95 distance-discount threshold.
Saved artifacts only; nothing is trained.  Outputs: outputs/20_*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pymatgen.core import Composition

from candidate_checks import OUT, SAVED, SEVEN, log, ssb, write_json  # noqa: I001 - sets sys.path for the release src
from extended_validation import clustered_metric_ci, point_metrics  # noqa: E402

MODEL = "catboost__composition_plus_context"
LI2 = "Li2Mn0.7Cr0.3O2.865F0.135"


def item_a() -> pd.DataFrame:
    oof = pd.read_csv(SAVED / "results" / "oof_canonical_composition_heldout_all_models.csv")
    oof = oof[oof["model"].eq(MODEL)].reset_index(drop=True)
    assign = pd.read_csv(SAVED / "results" / "known_control_prototype_assignment.csv")
    oof["proto_name"] = oof["canonical_key"].map(dict(zip(assign["canonical_key"], assign["proto_name"])))
    oof["formula"] = oof["canonical_key"].map(dict(zip(assign["canonical_key"], assign["canonical_formula"])))
    elements = oof["canonical_key"].str.split("|").apply(lambda t: {x.split(":")[0] for x in t})
    subsets = {
        "全體": np.ones(len(oof), bool),
        "Li2MnO3 模板": oof["proto_name"].eq("Li2MnO3").to_numpy(),
        "Li-rich NMC 模板": oof["proto_name"].eq("Li-rich-NMC").to_numpy(),
        "含 Cr": elements.apply(lambda s: "Cr" in s).to_numpy(),
        "含 F": elements.apply(lambda s: "F" in s).to_numpy(),
        "含 Cr 或 F": elements.apply(lambda s: bool({"Cr", "F"} & s)).to_numpy(),
    }
    rows = []
    for name, mask in subsets.items():
        part = oof[mask]
        m = point_metrics(part["y_true"].to_numpy(float), part["y_pred"].to_numpy(float))
        ci = clustered_metric_ci(part.reset_index(drop=True)) if part["canonical_key"].nunique() >= 3 else {}
        rows.append({"subset": name, "n_records": len(part), "n_compositions": part["canonical_key"].nunique(),
                     **m, **{f"{k}_ci95": v for k, v in ci.items()},
                     "mean_signed_error": float((part["y_pred"] - part["y_true"]).mean()),
                     "y_true_mean": float(part["y_true"].mean()), "y_pred_mean": float(part["y_pred"].mean())})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "20a_oof_subset_metrics.csv", index=False, encoding="utf-8-sig")
    # per-composition detail for the small subsets
    small = oof[subsets["Li2MnO3 模板"] | subsets["含 Cr 或 F"]]
    detail = small.groupby(["formula", "proto_name"], dropna=False).agg(
        n_records=("y_true", "size"), y_true_mean=("y_true", "mean"), y_pred_mean=("y_pred", "mean"),
        mae=("abs_error", "mean")).reset_index()
    detail.to_csv(OUT / "20a_cr_f_li2mno3_compositions.csv", index=False, encoding="utf-8-sig")
    return table


def item_b() -> pd.DataFrame:
    ev = ssb.CandidateEvaluator()
    self_distance, self_index = ev.model_nn.kneighbors(ev.train_scaled)
    train_d = self_distance[:, 1:].mean(axis=1)            # same as model_distance_p95's input
    if not np.isclose(np.percentile(train_d, 95), ev.model_distance_p95):
        raise RuntimeError("training 5NN distances do not reproduce model_distance_p95")
    keys = list(ev.unique_train.index)
    assign = pd.read_csv(SAVED / "results" / "known_control_prototype_assignment.csv")
    names = dict(zip(assign["canonical_key"], assign["canonical_formula"]))
    rows = []
    for formula in SEVEN + [LI2]:
        comp = Composition(formula)
        cand = {"comp": comp.get_el_amt_dict(), "Formula": formula, "canonical_key": ssb.canonical_key(comp),
                "proto_name": "Li2MnO3", "is_doped": True}
        out = ev.evaluate([cand]).iloc[0]
        X = pd.DataFrame([ev._composition_row(cand)]).reindex(columns=ev.comp_columns)
        scaled = ev.scaler.transform(ev.imputer.transform(X)[:, ev.nonconstant])
        dist, idx = ev.model_nn.kneighbors(scaled, n_neighbors=ssb.KNN_K)
        neighbours = [names[keys[i]] for i in idx[0]]
        rows.append({"Formula": formula, "model_distance": float(out["model_distance"]),
                     "percentile_in_training": float((train_d <= out["model_distance"]).mean() * 100),
                     "model_distance_p95": ev.model_distance_p95,
                     "below_p95": bool(out["model_distance"] <= ev.model_distance_p95),
                     "discount": float(out["discount"]), "y_pred": float(out["y_pred"]),
                     "nearest_5": "; ".join(f"{n} ({d:.3f})" for n, d in zip(neighbours, dist[0]))})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "20b_candidate_5nn.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"canonical_key": keys, "train_5nn_distance": train_d}).to_csv(
        OUT / "20b_training_5nn_distance.csv", index=False)
    write_json("20b_summary.json", {"n_training_compositions": len(train_d),
                                    "train_5nn_quantiles": {q: float(np.percentile(train_d, q))
                                                            for q in [5, 25, 50, 75, 90, 95, 99, 100]},
                                    "model_distance_p95": ev.model_distance_p95})
    return table


if __name__ == "__main__":
    log("20a"); print(item_a().round(3).to_string())
    log("20b"); print(item_b().round(4).to_string())
