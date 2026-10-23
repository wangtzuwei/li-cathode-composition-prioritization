"""21: G3 with charge-balanced, valence-dependent radii (analysis 21; see revision_analyses/README.md).

Original G3 (search_screen_benchmark.radius_mismatch_metrics) uses one fixed radius per element
(R6_LAYERED; R6_OLIVINE for olivine templates), e.g. Cr always 0.615 (Cr3+).  Here every redox-active
transition metal gets the average valence required by charge balance, and its radius is interpolated linearly
in valence between the Shannon octahedral radii of the neighbouring integer states.  G3 itself is unchanged:
concentration-weighted RMS of |r_i - r_host| / r_host over the TM_SET elements.

Valence rules
  fixed  Li, Na, K +1; Mg, Zn, Ca +2; Al, Ga, Sc, Y, La, Gd, Sm, Nd, B +3; Ti, Zr, Sn, Ge, Ru, Si +4; Nb, P +5;
         Mo, W +6; O -2; F -1
  redox  start from the reference valences Mn4+, Ni2+, Co3+, Cr3+, Fe3+, V3+, Cu2+.  The remaining charge is
         taken up one element at a time, each up to its limit before the next one moves:
           more positive charge needed (oxidation):  Ni (to 4), Co (to 4), Cr (to 6), V (to 5), Cu (to 3), Fe (to 4)
           less positive charge needed (reduction):  Mn (to 2), Co (to 2), Fe (to 2), V (to 2), Cr (to 2), Cu (to 1)
         e.g. NMC: Co3+ and Mn4+ fixed, Ni from charge balance; LiMn2O4: Mn3.5+; LiFePO4: Fe2+;
         Li2.1Mn0.7Cr0.3O2.865F0.135: Mn4+, Cr3.217+.
  radii (A, CN 6): Mn 2/3/4 = 0.83/0.645/0.53; Ni 2/3/4 = 0.69/0.56/0.48; Co 2/3/4 = 0.745/0.545/0.53;
         Cr 2/3/4/5/6 = 0.73/0.615/0.55/0.49/0.44; Fe 2/3/4 = 0.78/0.645/0.585; V 2/3/4/5 = 0.79/0.64/0.58/0.54;
         Cu 1/2/3 = 0.77/0.73/0.54.  The 2+ values are those of the original R6_OLIVINE table; the fixed-valence
         elements keep their R6_LAYERED radius (Ti 0.605 = Ti4+ in every template).
  host   A (main): host radius recomputed from the template base composition with the same rules
         (composition-weighted mean over TM_SET); B: original PROTO_HOST_R.
Calibration as in the release (np.quantile over the controls that pass G1 and G2) for the pooled threshold, and
as in 06 E for the structure-class thresholds.  The 20 reference TopK tables (nsga2__floor_on) are re-screened
with G1, G2, G4 unchanged and recurrence recomputed (>= 10 seeds, passing in every seed recovered).
Saved artifacts only; nothing is trained and no search is run.  Outputs: outputs/21_*.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from pymatgen.core import Composition

import candidate_checks as t
from site_limits_and_expanded_g3 import recurrence
from candidate_checks import OUT, SAVED, SEVEN, class_of, log, ssb, write_json

LI2 = "Li2Mn0.7Cr0.3O2.865F0.135"
FIXED = {"Li": 1, "Na": 1, "K": 1, "Mg": 2, "Zn": 2, "Ca": 2, "Al": 3, "Ga": 3, "Sc": 3, "Y": 3, "La": 3, "Gd": 3,
         "Sm": 3, "Nd": 3, "B": 3, "Ti": 4, "Zr": 4, "Sn": 4, "Ge": 4, "Ru": 4, "Si": 4, "Nb": 5, "P": 5, "Mo": 6,
         "W": 6, "O": -2, "F": -1}
REFERENCE = {"Mn": 4, "Ni": 2, "Co": 3, "Cr": 3, "Fe": 3, "V": 3, "Cu": 2}
OXIDATION = [("Ni", 4), ("Co", 4), ("Cr", 6), ("V", 5), ("Cu", 3), ("Fe", 4)]
REDUCTION = [("Mn", 2), ("Co", 2), ("Fe", 2), ("V", 2), ("Cr", 2), ("Cu", 1)]
RADII = {
    "Mn": {2: 0.83, 3: 0.645, 4: 0.53}, "Ni": {2: 0.69, 3: 0.56, 4: 0.48}, "Co": {2: 0.745, 3: 0.545, 4: 0.53},
    "Cr": {2: 0.73, 3: 0.615, 4: 0.55, 5: 0.49, 6: 0.44}, "Fe": {2: 0.78, 3: 0.645, 4: 0.585},
    "V": {2: 0.79, 3: 0.64, 4: 0.58, 5: 0.54}, "Cu": {1: 0.77, 2: 0.73, 3: 0.54},
}
TOL = 1e-9


def valences(formula: str) -> tuple[dict[str, float], float]:
    """Average valence of every redox element; second value = charge left unassigned (0 when balanced)."""
    comp = Composition(formula).get_el_amt_dict()
    unknown = set(comp) - set(FIXED) - set(REFERENCE)
    if unknown:
        raise KeyError(f"{formula}: no valence rule for {sorted(unknown)}")
    vals = {el: float(REFERENCE[el]) for el in comp if el in REFERENCE}
    residual = -sum(FIXED[el] * x for el, x in comp.items() if el in FIXED) - sum(vals[el] * comp[el] for el in vals)
    order = OXIDATION if residual > 0 else REDUCTION
    for el, limit in order:
        if abs(residual) < TOL:
            break
        if el not in vals:
            continue
        room = (limit - vals[el]) * comp[el]          # signed charge this element can still take
        step = max(min(residual, room), 0.0) if residual > 0 else min(max(residual, room), 0.0)
        vals[el] += step / comp[el]
        residual -= step
    return vals, residual


def radius(el: str, valence: float | None) -> float:
    if el not in RADII:
        return ssb.R6_LAYERED[el]
    table = RADII[el]
    lo = max(k for k in table if k <= valence + TOL)
    hi = min(k for k in table if k >= valence - TOL)
    return table[lo] if lo == hi else table[lo] + (table[hi] - table[lo]) * (valence - lo) / (hi - lo)


def tm_radii(formula: str) -> dict[str, float]:
    vals, _ = valences(formula)
    comp = Composition(formula).get_el_amt_dict()
    return {el: radius(el, vals.get(el)) for el in comp if el in ssb.TM_SET}


PROTOS = {p["name"]: p for p in ssb.prototypes()}


def host_radius(proto: str) -> float:
    base = Composition(PROTOS[proto]["base_comp"])
    comp = base.get_el_amt_dict()
    r = tm_radii(base.formula)
    total = sum(comp[el] for el in r)
    return sum(comp[el] * r[el] for el in r) / total


HOST_A = {name: host_radius(name) for name in PROTOS}
HOST_B = dict(ssb.PROTO_HOST_R)


def g3_valence(formula: str, proto: str, host: dict[str, float]) -> float:
    if proto not in host or not isinstance(proto, str):
        return math.nan
    comp = Composition(formula).get_el_amt_dict()
    r = tm_radii(formula)
    h = host[proto]
    num = sum(comp[el] * ((r[el] - h) / h) ** 2 for el in r if comp[el] > 0)
    den = sum(comp[el] for el in r if comp[el] > 0)
    return math.sqrt(num / den) if den > 0 else math.nan


def describe(formula: str) -> str:
    vals, residual = valences(formula)
    text = ", ".join(f"{el}{v:+.3f}" for el, v in sorted(vals.items()))
    return text + (f" (unbalanced {residual:+.3f})" if abs(residual) > 1e-6 else "")


def thresholds(controls: pd.DataFrame, column: str) -> dict[str, dict[str, float]]:
    reached = controls[controls["pass_g1_g2"].astype(bool)]
    out = {"pooled": {"P90": float(np.quantile(reached[column], 0.90)),
                      "P95": float(np.quantile(reached[column], 0.95)), "n": int(len(reached))}}
    for cls, part in reached.groupby("class"):
        v = part[column].to_numpy(float)
        out[cls] = {"P90": float(np.quantile(v, 0.90)) if len(v) >= 10 else math.nan,
                    "P95": float(np.quantile(v, 0.95)) if len(v) >= 10 else math.nan, "n": int(len(v))}
    return out


def main() -> None:
    # ---- controls
    controls = pd.read_csv(SAVED / "results" / "known_control_screened.csv")
    controls["class"] = controls["proto_name"].map(class_of)
    controls["G3_original"] = controls["radius_mismatch"]
    check = [abs(ssb.radius_mismatch_metrics(f, p)[0] - g) < 1e-12
             for f, p, g in zip(controls["Formula"], controls["proto_name"], controls["G3_original"])]
    if not all(check):
        raise RuntimeError("saved control G3 not reproduced")
    controls["valences"] = controls["Formula"].map(describe)
    controls["G3_valence_A"] = [g3_valence(f, p, HOST_A) for f, p in zip(controls["Formula"], controls["proto_name"])]
    controls["G3_valence_B"] = [g3_valence(f, p, HOST_B) for f, p in zip(controls["Formula"], controls["proto_name"])]
    controls[["proto_name", "class", "Formula", "pass_g1_g2", "valences", "G3_original", "G3_valence_A",
              "G3_valence_B"]].to_csv(OUT / "21_controls_g3.csv", index=False, encoding="utf-8-sig")
    thr = {"original": thresholds(controls, "G3_original"), "valence_A": thresholds(controls, "G3_valence_A"),
           "valence_B": thresholds(controls, "G3_valence_B")}
    p95_saved, p90_saved = t.p95_p90()
    if not (np.isclose(thr["original"]["pooled"]["P95"], p95_saved) and np.isclose(thr["original"]["pooled"]["P90"], p90_saved)):
        raise RuntimeError("original pooled thresholds not reproduced")
    log(f"thresholds {thr}")

    # ---- re-screen the 20 reference TopK tables
    rows = t.reference_rows()
    rows["class"] = rows["proto_name"].map(class_of)
    rows["G3_original"] = pd.to_numeric(rows["radius_mismatch"], errors="coerce")
    pairs = rows[["Formula", "proto_name"]].drop_duplicates()
    cache_a = {(f, p): g3_valence(f, p, HOST_A) for f, p in zip(pairs["Formula"], pairs["proto_name"])}
    cache_b = {(f, p): g3_valence(f, p, HOST_B) for f, p in zip(pairs["Formula"], pairs["proto_name"])}
    rows["G3_valence_A"] = [cache_a[(f, p)] for f, p in zip(rows["Formula"], rows["proto_name"])]
    rows["G3_valence_B"] = [cache_b[(f, p)] for f, p in zip(rows["Formula"], rows["proto_name"])]
    unbalanced = sum(abs(valences(f)[1]) > 1e-6 for f in rows["Formula"].unique())
    g124 = rows["g1_pass"].astype(bool) & rows["g2_pass"].astype(bool) & rows["g4_pass"].astype(bool)
    results, flags = {}, []
    for version in ["original", "valence_A", "valence_B"]:
        col = "G3_original" if version == "original" else f"G3_{version}"
        for scope in ["pooled", "class"]:
            for q in ["P95", "P90"]:
                if scope == "pooled":
                    limit = pd.Series(thr[version]["pooled"][q], index=rows.index)
                else:
                    limit = rows["class"].map({c: v[q] for c, v in thr[version].items() if c != "pooled"})
                flag = f"{version}__{scope}_{q}"
                rows[flag] = g124 & rows[col].le(limit + TOL)
                flags.append(flag)
                results[flag] = recurrence(rows, flag)
    if results["original__pooled_P95"]["seven_retained"] != sorted(SEVEN):
        raise RuntimeError("original pooled P95 recurrence does not reproduce the 7 candidates")
    rows[["seed", "canonical_key", "Formula", "proto_name", "class", "G3_original", "G3_valence_A",
          "G3_valence_B"] + flags].to_csv(OUT / "21_reference_rows.csv", index=False)

    # ---- the 7 candidates and the Li = 2 member
    seven = []
    for f in SEVEN + [LI2]:
        row = {"Formula": f, "valences": describe(f), "G3_original": ssb.radius_mismatch_metrics(f, "Li2MnO3")[0],
               "G3_valence_A": g3_valence(f, "Li2MnO3", HOST_A), "G3_valence_B": g3_valence(f, "Li2MnO3", HOST_B)}
        for version, col in [("original", "G3_original"), ("valence_A", "G3_valence_A"), ("valence_B", "G3_valence_B")]:
            for scope in ["pooled", "layered"]:
                for q in ["P95", "P90"]:
                    row[f"pass_{version}_{scope}_{q}"] = bool(row[col] <= thr[version][scope][q] + TOL)
        seven.append(row)
    seven = pd.DataFrame(seven)
    seven.to_csv(OUT / "21_seven_g3.csv", index=False, encoding="utf-8-sig")
    hosts = pd.DataFrame({"proto_name": list(PROTOS), "class": [class_of(p) for p in PROTOS],
                          "host_original": [HOST_B[p] for p in PROTOS], "host_valence": [HOST_A[p] for p in PROTOS],
                          "base_valences": [describe(Composition(PROTOS[p]["base_comp"]).formula) for p in PROTOS]})
    base_formula = {p: Composition(PROTOS[p]["base_comp"]).formula for p in PROTOS}
    hosts["base_G3_original"] = [ssb.radius_mismatch_metrics(base_formula[p], p)[0] for p in PROTOS]
    hosts["base_G3_valence_A"] = [g3_valence(base_formula[p], p, HOST_A) for p in PROTOS]
    hosts["base_G3_valence_B"] = [g3_valence(base_formula[p], p, HOST_B) for p in PROTOS]
    hosts.to_csv(OUT / "21_hosts.csv", index=False, encoding="utf-8-sig")
    write_json("21_summary.json", {"thresholds": thr, "rescreen": results, "hosts": hosts.to_dict(orient="records"),
                                   "seven": seven.to_dict(orient="records"),
                                   "unbalanced_topk_formulas": int(unbalanced),
                                   "unbalanced_controls": int(sum(abs(valences(f)[1]) > 1e-6 for f in controls["Formula"]))})
    log("done")


if __name__ == "__main__":
    main()
