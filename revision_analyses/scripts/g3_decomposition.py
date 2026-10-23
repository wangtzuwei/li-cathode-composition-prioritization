"""G3 decomposition and the Li2Mn(1-x)CrxO3 G3 curve (analysis 18; see revision_analyses/README.md).

G3 (search_screen_benchmark.radius_mismatch_metrics) is the concentration-weighted RMS of the relative
radius mismatch |r_i - r_host| / r_host over the elements in TM_SET only:
    G3 = sqrt( sum_i n_i * d_i^2 / sum_i n_i ),  i in TM_SET
so each element's share of G3^2 is n_i * d_i^2 / sum_j n_j * d_j^2.  Li, O and F are not in TM_SET.
1  per-element contributions for the 7 candidates and the Li = 2 member (template Li2MnO3), plus G3 of
   variants with F removed and Li changed, computed with the original function
2  G3 of Li2Mn(1-x)CrxO3, x = 0 ... 0.5 (step 0.05), and the exact crossing of the layered-class P95 and
   the pooled P95; figure with the 7 candidates, Li1.2Cr0.4Mn0.4O2 and the Zhang 2013 series
Outputs: outputs/18_*, outputs/figures/18_g3_curve.png.  Nothing is trained.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import candidate_checks as t  # noqa: E402
from candidate_checks import OUT, SEVEN, ssb, write_json  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "outputs" / "figures" / "18_g3_curve.png"
LI2 = "Li2Mn0.7Cr0.3O2.865F0.135"
PROTO = "Li2MnO3"
LAYERED_P95 = 0.0659489882206248
POOLED_P95 = t.p95_p90()[0]
ZHANG = {"A（x = 1/60）": "Li1.03333Mn0.316667Ni0.25Cr0.4O2", "B（x = 1/30）": "Li1.06667Mn0.333333Ni0.2Cr0.4O2",
         "C（x = 1/20）": "Li1.1Mn0.35Ni0.15Cr0.4O2", "D（x = 1/15）": "Li1.13333Mn0.366667Ni0.1Cr0.4O2",
         "E（x = 1/12）": "Li1.16667Mn0.383333Ni0.05Cr0.4O2"}
LCMO = "Li1.2Cr0.4Mn0.4O2"


def contributions(formula: str, proto: str) -> dict[str, object]:
    comp = Composition(formula).get_el_amt_dict()
    host = ssb.PROTO_HOST_R[proto]
    radii = ssb.R6_OLIVINE if ssb.PROTO_FAMILY.get(proto) == "olivine" else ssb.R6_LAYERED
    terms = {e: n * ((radii[e] - host) / host) ** 2 for e, n in comp.items() if e in ssb.TM_SET and e in radii}
    total_n = sum(n for e, n in comp.items() if e in ssb.TM_SET and e in radii)
    total = sum(terms.values())
    g3 = math.sqrt(total / total_n)
    check = ssb.radius_mismatch_metrics(formula, proto)[0]
    if abs(g3 - check) > 1e-12:
        raise RuntimeError(f"decomposition does not reproduce G3 for {formula}")
    out = {"Formula": formula, "template": proto, "host_radius": host, "G3": g3, "tm_total": total_n,
           "Cr_fraction_of_TM": comp.get("Cr", 0.0) / total_n}
    for e in ["Mn", "Cr", "Ni", "Co", "Fe"]:
        if e in comp:
            out[f"{e}_amount"] = comp[e]
            out[f"{e}_rel_mismatch"] = (radii[e] - host) / host
            out[f"{e}_share_of_G3sq"] = terms[e] / total if total else 0.0
    for e in ["Li", "O", "F"]:
        out[f"{e}_amount"] = comp.get(e, 0.0)
    return out


def main() -> None:
    members = SEVEN + [LI2]
    rows = [contributions(f, PROTO) for f in members]
    dec = pd.DataFrame(rows)
    # variants: same Mn/Cr, F removed (O = 3) and Li changed, to show the effect of each substitution
    variants = []
    for f in members:
        amt = Composition(f).get_el_amt_dict()
        mn, cr = amt["Mn"], amt["Cr"]
        for label, li, o, fl in [("as is", amt["Li"], amt["O"], amt.get("F", 0.0)), ("no F (O = 3)", amt["Li"], 3.0, 0.0),
                                 ("Li = 2, no F", 2.0, 3.0, 0.0), ("no Cr (Mn = 1)", amt["Li"], amt["O"], amt.get("F", 0.0))]:
            m, c = (1.0, 0.0) if label.startswith("no Cr") else (mn, cr)
            formula = f"Li{li:g}Mn{m:g}" + (f"Cr{c:g}" if c else "") + f"O{o:g}" + (f"F{fl:g}" if fl else "")
            variants.append({"member": f, "variant": label, "Formula": formula,
                             "G3": ssb.radius_mismatch_metrics(formula, PROTO)[0]})
    dec.to_csv(OUT / "18_g3_decomposition.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(variants).to_csv(OUT / "18_g3_variants.csv", index=False, encoding="utf-8-sig")

    xs = np.round(np.arange(0, 0.5001, 0.05), 2)
    curve = pd.DataFrame({"x": xs, "Formula": [f"Li2Mn{1 - x:g}Cr{x:g}O3" if x else "Li2Mn1O3" for x in xs]})
    curve["G3"] = [ssb.radius_mismatch_metrics(f, PROTO)[0] for f in curve["Formula"]]
    curve["above_layered_P95"] = curve["G3"] > LAYERED_P95
    curve["above_pooled_P95"] = curve["G3"] > POOLED_P95
    curve.to_csv(OUT / "18_g3_curve.csv", index=False, encoding="utf-8-sig")
    d_cr = (ssb.R6_LAYERED["Cr"] - ssb.PROTO_HOST_R[PROTO]) / ssb.PROTO_HOST_R[PROTO]
    cross_layered = (LAYERED_P95 / d_cr) ** 2  # G3 = d_cr * sqrt(x) for Li2Mn(1-x)CrxO3
    cross_pooled = (POOLED_P95 / d_cr) ** 2

    refs = [contributions(LCMO, PROTO) | {"label": "Li1.2Cr0.4Mn0.4O2", "assigned": ssb.assign_known_prototype(LCMO)[0],
                                          "G3_assigned": ssb.radius_mismatch_metrics(LCMO, ssb.assign_known_prototype(LCMO)[0])[0]}]
    for lab, f in ZHANG.items():
        proto = ssb.assign_known_prototype(f)[0]
        refs.append(contributions(f, PROTO) | {"label": f"Zhang 2013 {lab}", "assigned": proto,
                                               "G3_assigned": ssb.radius_mismatch_metrics(f, proto)[0]})
    refs = pd.DataFrame(refs)
    refs.to_csv(OUT / "18_g3_references.csv", index=False, encoding="utf-8-sig")
    plot(curve, dec, refs)
    write_json("18_summary.json", {"Cr_relative_mismatch_vs_Li2MnO3_host": d_cr,
                                   "Mn_relative_mismatch": (ssb.R6_LAYERED["Mn"] - ssb.PROTO_HOST_R[PROTO]) / ssb.PROTO_HOST_R[PROTO],
                                   "layered_P95": LAYERED_P95, "pooled_P95": POOLED_P95,
                                   "x_crossing_layered_P95": cross_layered, "x_crossing_pooled_P95": cross_pooled,
                                   "first_grid_x_above_layered_P95": float(curve.loc[curve["above_layered_P95"], "x"].min()),
                                   "first_grid_x_above_pooled_P95": (float(curve.loc[curve["above_pooled_P95"], "x"].min())
                                                                     if curve["above_pooled_P95"].any() else None)})


def plot(curve: pd.DataFrame, dec: pd.DataFrame, refs: pd.DataFrame) -> None:
    FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "axes.edgecolor": "#52514e"})
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    xf = np.linspace(0, 0.5, 201)
    d_cr = (ssb.R6_LAYERED["Cr"] - ssb.PROTO_HOST_R[PROTO]) / ssb.PROTO_HOST_R[PROTO]
    ax.plot(xf, d_cr * np.sqrt(xf), color="#2a78d6", linewidth=2, label="Li$_2$Mn$_{1-x}$Cr$_x$O$_3$ (Li$_2$MnO$_3$ template)")
    ax.plot(curve["x"], curve["G3"], "o", color="#2a78d6", markersize=4)
    for y, text in [(LAYERED_P95, "layered-class P95 = 0.0659"), (POOLED_P95, "pooled P95 = 0.1087")]:
        ax.axhline(y, color="#52514e", linewidth=0.9, linestyle="--")
        ax.text(0.005, y + 0.0015, text, color="#52514e", fontsize=8, va="bottom")
    cand = dec.drop_duplicates("Cr_fraction_of_TM")
    ax.plot(dec["Cr_fraction_of_TM"], dec["G3"], "D", color="#eb6834", markersize=8, markeredgecolor="#fcfcfb",
            markeredgewidth=1.5, label="7 candidates and Li = 2 member", zorder=5)
    for r in cand.itertuples():
        offset = (10, -14) if r.Cr_fraction_of_TM > 0.29 else (-78, 6)
        ax.annotate(f"Cr/TM = {r.Cr_fraction_of_TM:.2f}", (r.Cr_fraction_of_TM, r.G3), textcoords="offset points",
                    xytext=offset, fontsize=7.5, color="#52514e")
    x_cross = (LAYERED_P95 / d_cr) ** 2
    ax.plot([x_cross, x_cross], [0, LAYERED_P95], color="#52514e", linewidth=0.8, linestyle=":")
    ax.annotate(f"x = {x_cross:.3f}", (x_cross, 0.004), textcoords="offset points", xytext=(4, 0),
                fontsize=7.5, color="#52514e")
    lc = refs[refs["label"].eq("Li1.2Cr0.4Mn0.4O2")].iloc[0]
    ax.plot(lc["Cr_fraction_of_TM"], lc["G3"], "s", color="#1baf7a", markersize=8, markeredgecolor="#fcfcfb",
            markeredgewidth=1.5, label="Li$_{1.2}$Cr$_{0.4}$Mn$_{0.4}$O$_2$ (Ammundsen 2002 / [63])", zorder=5)
    z = refs[refs["label"].str.startswith("Zhang")]
    ax.plot(z["Cr_fraction_of_TM"], z["G3"], "^", color="#eda100", markersize=8, markeredgecolor="#fcfcfb",
            markeredgewidth=1.5, label="Zhang 2013 A–E (Cr$_{0.4}$, with Ni)", zorder=5)
    ax.set_xlim(-0.01, 0.52)
    ax.set_xlabel("Cr fraction of transition-metal sites, x = Cr / (Cr + Mn [+ Ni])")
    ax.set_ylabel("G3 (weighted RMS radius mismatch)")
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.text(0.01, 0.01, "All points use the Li2MnO3 template (host radius 0.530 Å). Li, O and F do not enter G3.",
             fontsize=7.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG, dpi=200, facecolor="#fcfcfb")
    plt.close(fig)


if __name__ == "__main__":
    main()
