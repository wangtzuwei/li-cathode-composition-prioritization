"""Family-holdout leakage with distance thresholds (analysis 06c; see revision_analyses/README.md).

For each family (layered, olivine, spinel):
  - nearest-neighbour distance of every member to the other members of the same
    family (atomic-fraction Manhattan distance, as in 06 F / 06b item 4);
  - thresholds = P5, P10, P25, P50 of that distribution (np.percentile, linear);
  - every untemplated composition whose minimum distance to the family is
    <= threshold is removed from that family's training set (ties included);
  - the family holdout is rerun with the original settings
    (extended_validation.leave_one_family_out_validation: CB_PARAMS,
    thread_count 1, random_seed 43000 + family index).
New models go to models_local/family_leakage_distance/, tables to outputs/06c_*.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

import candidate_checks as t
from candidate_checks import OUT, SAVED, log, write_json

MODEL_OUT = t.HERE.parent / "models_local" / "family_leakage_distance"
FAMILIES = ["layered", "olivine", "spinel"]
PERCENTILES = [5, 10, 25, 50]


def main() -> None:
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    t.MODEL_OUT = MODEL_OUT  # family_holdout saves into candidate_checks.MODEL_OUT
    features, y, raw = t.training_table()
    assign = pd.read_csv(SAVED / "results" / "known_control_prototype_assignment.csv")
    names = assign.set_index("canonical_key")["canonical_formula"]
    templated = assign.dropna(subset=["proto_name"]).set_index("canonical_key")["proto_name"].astype(str)
    family_by_key = templated.map(t.STRUCTURAL_FAMILY)
    untemplated = assign[assign["proto_name"].isna()][["canonical_key", "canonical_formula", "n_records"]]

    nn_rows, dist_rows, thresholds = [], [], {}
    for family in FAMILIES:
        members = list(family_by_key[family_by_key.eq(family)].index)
        M = np.array([t.frac_vector(k) for k in members])
        within = np.abs(M[:, None, :] - M[None, :, :]).sum(axis=2)
        np.fill_diagonal(within, np.inf)
        nn = within.min(axis=1)
        for k, d in zip(members, nn):
            nn_rows.append({"family": family, "member": names[k], "proto_name": templated[k], "nn_distance": float(d)})
        thresholds[family] = {f"P{p}": float(np.percentile(nn, p)) for p in PERCENTILES}
        thresholds[family].update({"n_members": len(members), "min": float(nn.min()), "max": float(nn.max()),
                                   "median": float(np.median(nn)), "n_zero": int((nn == 0).sum())})
        for u in untemplated.itertuples(index=False):
            d = np.abs(M - t.frac_vector(u.canonical_key)).sum(axis=1)
            j = int(np.argmin(d))
            dist_rows.append({"family": family, "canonical_key": u.canonical_key, "formula": u.canonical_formula,
                              "n_records": int(u.n_records), "min_distance": float(d[j]),
                              "nearest_member": names[members[j]]})
    pd.DataFrame(nn_rows).to_csv(OUT / "06c_within_family_nn.csv", index=False)
    dist = pd.DataFrame(dist_rows).sort_values(["family", "min_distance", "formula"])
    dist.to_csv(OUT / "06c_untemplated_min_distance.csv", index=False)

    runs = {}
    for label, p in [("P00_original", None)] + [(f"P{p:02d}", p) for p in PERCENTILES]:
        drop = {}
        for family in FAMILIES:
            if p is None:
                drop[family] = []
                continue
            thr = thresholds[family][f"P{p}"]
            part = dist[dist["family"].eq(family) & (dist["min_distance"] <= thr + 1e-12)]
            drop[family] = part["canonical_key"].tolist()
        _, summary = t.family_holdout(features, y, raw, family_by_key, drop, f"c_{label}")
        (OUT / f"06F_c_{label}_predictions.csv").replace(OUT / f"06c_{label}_predictions.csv")
        runs[label] = {**summary, "percentile": p,
                       "thresholds": {f: (thresholds[f][f"P{p}"] if p else None) for f in FAMILIES},
                       "removed": {f: dist[dist["canonical_key"].isin(drop[f]) & dist["family"].eq(f)][
                           ["formula", "n_records", "min_distance", "nearest_member"]].to_dict(orient="records")
                                   for f in FAMILIES}}
        log(f"{label}: overall R2 {summary['overall']['r2']:.4f}")
    saved = json.loads((SAVED / "results" / "leave_one_family_out_validation.json").read_text(encoding="utf-8"))
    write_json("06c_summary.json", {"thresholds": thresholds, "saved_overall_r2": saved["overall"]["r2"], "runs": runs})


if __name__ == "__main__":
    start = time.time()
    main()
    log(f"done in {time.time() - start:.0f} s")
