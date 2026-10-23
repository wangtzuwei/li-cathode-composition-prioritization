"""15: site-occupancy calibration from the positive controls; 16: layered G3 with expanded controls.

15  cation/anion ratio of the 112 positive controls by structure class -> P95 and maximum as limits;
    applied after G1-G4 (P95) to the 20 reference TopK tables, recurrence recomputed.
16  synthesized layered materials from analysis 11 (outputs/11_external_g3_combined.csv) are added to the
    layered controls; G1/G2 as in the original calibration (only controls passing G1 and G2 enter G3),
    layered P90/P95 recomputed with np.quantile; olivine and spinel keep their class thresholds from 06 E;
    the 20 seeds are re-screened and recurrence recomputed.  Two versions: with and without Zhang 2013.
Saved artifacts only; nothing is trained and no search is run.  Outputs: outputs/15_*, outputs/16_*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pymatgen.core import Composition

import candidate_checks as t
from candidate_checks import OUT, SAVED, SEVEN, class_of, log, ssb, write_json

ORIGINAL = set(pd.read_csv(SAVED / "results" / "consensus_candidates.csv")["Formula"])
TOL = 1e-9


def ratio(formula: str) -> float:
    return t.cation_anion(formula)


def recurrence(rows: pd.DataFrame, flag: str) -> dict[str, object]:
    g124 = rows["g1_pass"].astype(bool) & rows["g2_pass"].astype(bool) & rows["g4_pass"].astype(bool)
    records = []
    for key, part in rows.groupby("canonical_key", sort=True):
        seeds = part["seed"].nunique()
        passing = part.loc[part[flag], "seed"].nunique()
        records.append({"canonical_key": key, "Formula": part["Formula"].iloc[0], "proto_name": part["proto_name"].iloc[0],
                        "topk_seed_count": seeds, "pass_seed_count": passing,
                        "consensus": seeds >= 10 and passing == seeds and bool(g124[part.index].all())})
    union = pd.DataFrame(records)
    cons = union[union["consensus"]].sort_values("topk_seed_count", ascending=False)
    per_seed = rows.groupby("seed")[flag].sum()
    return {"union": int(len(union)), "consensus_n": int(len(cons)),
            "consensus": cons[["Formula", "proto_name", "topk_seed_count"]].to_dict(orient="records"),
            "seven_retained": sorted(ORIGINAL & set(cons["Formula"])),
            "seven_dropped": sorted(ORIGINAL - set(cons["Formula"])),
            "new": sorted(set(cons["Formula"]) - ORIGINAL),
            "per_seed_mean": float(per_seed.mean()), "per_seed_min": int(per_seed.min()),
            "per_seed_max": int(per_seed.max())}


def item_15() -> None:
    log("15")
    controls = pd.read_csv(SAVED / "results" / "known_control_screened.csv")
    controls["class"] = controls["proto_name"].map(class_of)
    controls["ratio"] = controls["Formula"].map(ratio)
    limits = {}
    for cls, part in controls.groupby("class"):
        v = part["ratio"].to_numpy(float)
        limits[cls] = {"n": int(len(v)), "P95": float(np.quantile(v, 0.95)), "max": float(v.max()),
                       "min": float(v.min()), "median": float(np.median(v))}
    rows = t.reference_rows()
    rows["class"] = rows["proto_name"].map(class_of)
    rows["ratio"] = rows["Formula"].map(ratio)
    out = {"limits": limits}
    for basis in ["P95", "max"]:
        lim = rows["class"].map({c: v[basis] for c, v in limits.items()})
        rows[f"site_ok_{basis}"] = rows["ratio"] <= lim + TOL
        rows[f"final_{basis}"] = rows["p95_final"] & rows[f"site_ok_{basis}"]
        out[basis] = recurrence(rows, f"final_{basis}")
    out["original"] = recurrence(rows, "p95_final")
    seven = []
    for f in SEVEN:
        r = ratio(f)
        seven.append({"Formula": f, "ratio": r, "pass_P95": bool(r <= limits["layered"]["P95"] + TOL),
                      "pass_max": bool(r <= limits["layered"]["max"] + TOL)})
    out["seven"] = seven
    write_json("15_summary.json", out)


EXPANDED_SOURCES = {  # synthesized layered materials in 11_external_g3_combined.csv (source labels as stored)
    "Lee et al. 2015", "Wang et al. 2024", "Kucuk et al. 2026", "Ammundsen 2002", "[63] Pan et al. 2002",
    "Zhang 2013", "Fan 2024", "Hua 2019", "Marie 2024", "Ming 2018", "[65] Huang et al. 2021",
    "[64] Dong et al. 2015", "Nakayama 2020",
}
EXCLUDED_SOURCES = {"Im et al. 2019": "spinel–layered 複合材料", "Chambers 2025": "DRX（非層狀）",
                    "Blumenhofer 2023": "DRX（非層狀）", "Moździerz 2025": "部分無序 Li-excess（Cr/Ti）",
                    "7 candidates": "候選本身"}


def item_16() -> None:
    log("16")
    ext = pd.read_csv(OUT / "11_external_g3_combined.csv", encoding="utf-8-sig")
    controls = pd.read_csv(SAVED / "results" / "known_control_screened.csv")
    control_keys = set(controls["canonical_key"])
    new = []
    for r in ext.itertuples(index=False):
        if r.source not in EXPANDED_SOURCES:
            continue
        comp = Composition(r.Formula)
        key = ssb.canonical_key(comp)
        proto, rule = ssb.assign_known_prototype(r.Formula)
        g1 = bool(ssb.gate1_ionicity(r.Formula)[0])
        g2 = bool(ssb.gate2_li_tm(r.Formula, proto)[0]) if proto else False
        rms = ssb.radius_mismatch_metrics(r.Formula, proto)[0] if proto else np.nan
        new.append({"source": r.source, "sample": r.sample, "Formula": r.Formula, "canonical_key": key,
                    "assigned_template": proto, "class": class_of(proto) if proto else None, "G1": g1, "G2": g2,
                    "G3": rms, "already_a_control": key in control_keys, "zhang2013": r.source == "Zhang 2013"})
    new = pd.DataFrame(new).drop_duplicates("canonical_key").reset_index(drop=True)
    # [64] Dong 2015: the same samples are already controls under the paper's own oxygen-vacancy formulas
    # (Li2MnO2.9390F0.015 etc.; see analysis 09 section 1.3), so their nominal formulas are not added again.
    new["excluded_reason"] = np.where(new["already_a_control"], "組成已在 112 個對照組中", "")
    dong = new["source"].eq("[64] Dong et al. 2015") & new["excluded_reason"].eq("")
    new.loc[dong, "already_a_control"] = True
    new.loc[dong, "excluded_reason"] = "同一批樣品已以含氧空缺的化學式列在對照組中"
    new.to_csv(OUT / "16_added_layered_materials.csv", index=False, encoding="utf-8-sig")
    reached = controls[controls["pass_g1_g2"].astype(bool)].copy()
    reached["class"] = reached["proto_name"].map(class_of)
    base_layered = reached[reached["class"].eq("layered")]["radius_mismatch"].to_numpy(float)
    e_thr = pd.read_csv(OUT / "06E_thresholds.csv").query("level == 'class'").set_index("name")
    rows = t.reference_rows()
    rows["class"] = rows["proto_name"].map(class_of)
    radius = pd.to_numeric(rows["radius_mismatch"], errors="coerce")
    g124 = rows["g1_pass"].astype(bool) & rows["g2_pass"].astype(bool) & rows["g4_pass"].astype(bool)
    out = {"n_added_candidates": int(len(new))}
    for version, mask in [("with_zhang", np.ones(len(new), bool)), ("without_zhang", ~new["zhang2013"].to_numpy())]:
        add = new[mask & new["G1"] & new["G2"] & ~new["already_a_control"] & new["class"].eq("layered")]
        values = np.concatenate([base_layered, add["G3"].to_numpy(float)])
        thr = {"P90": float(np.quantile(values, 0.90)), "P95": float(np.quantile(values, 0.95)),
               "n": int(len(values)), "max": float(values.max())}
        res = {"layered": thr, "added": add[["source", "sample", "Formula", "G3"]].to_dict(orient="records")}
        for q in ["P95", "P90"]:
            cls_thr = {"layered": thr[q], "olivine": float(e_thr.loc["olivine", q]), "spinel": float(e_thr.loc["spinel", q])}
            rows[f"f_{version}_{q}"] = g124 & radius.le(rows["class"].map(cls_thr))
            res[q] = {"class_thresholds": cls_thr, **recurrence(rows, f"f_{version}_{q}")}
        seven = []
        for f in SEVEN:
            g3 = ssb.radius_mismatch_metrics(f, "Li2MnO3")[0]
            seven.append({"Formula": f, "G3": g3, "pass_P95": bool(g3 <= thr["P95"]), "pass_P90": bool(g3 <= thr["P90"]),
                          "n_controls_below": int((values < g3).sum())})
        res["seven"] = seven
        out[version] = res
    write_json("16_summary.json", out)


if __name__ == "__main__":
    item_15()
    item_16()
