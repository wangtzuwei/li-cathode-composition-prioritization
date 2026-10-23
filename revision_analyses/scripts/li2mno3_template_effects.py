"""Li2MnO3-template groundwork (analysis 12; see revision_analyses/README.md).

a  every training record traced to Dong 2015 ([64], 10.1039/C4TA02924K) by 09, per x and condition
b  model F effect and Cr effect on four host compositions (full model + 100 saved bootstrap models,
   298.15 K, 0.5 C, cycle 1; nothing trained):
     F replaces a fraction f of the O (anion total kept), f = 0 ... 0.10 in steps of 0.01;
     Cr replaces c Mn per formula unit (Co for LiCoO2), c = 0 ... 0.30 in steps of 0.05
c  five nearest training compositions of the 08 samples with direction disagreements (Kucuk 2026,
   Lee 2015), and the training records of Li1.2Ni0.2Mn0.6O2
Outputs: outputs/12_*, outputs/figures/12_F_Cr_effects.png.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pymatgen.core import Composition

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import followup_checks as f6  # noqa: E402
from composition_counterparts import load_models  # noqa: E402
from literature_pairs import features, load_samples  # noqa: E402
from candidate_checks import OUT, SAVED, log, ssb, write_json  # noqa: E402

FIG_DIR = f6.LITERATURE.parent.parent / "outputs" / "figures"
DONG = "10.1039/C4TA02924K"
HOSTS = [  # label, (Li, {TM: amount}, O), substituted TM
    ("Li2MnO3", 2.0, {"Mn": 1.0}, 3.0, "Mn"),
    ("Li1.2Ni0.2Mn0.6O2", 1.2, {"Ni": 0.2, "Mn": 0.6}, 2.0, "Mn"),
    ("Li1.2Mn0.54Ni0.13Co0.13O2", 1.2, {"Mn": 0.54, "Ni": 0.13, "Co": 0.13}, 2.0, "Mn"),
    ("LiCoO2", 1.0, {"Co": 1.0}, 2.0, "Co"),
]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4 (dataviz reference palette)
F_GRID = np.round(np.arange(0, 0.1001, 0.01), 2)
CR_GRID = np.round(np.arange(0, 0.3001, 0.05), 2)


def fmt(x: float) -> str:
    return f"{x:.6g}"


def formula(li: float, tms: dict[str, float], o: float, f: float = 0.0) -> str:
    parts = [f"Li{fmt(li)}"] + [f"{el}{fmt(v)}" for el, v in tms.items() if v > 1e-12]
    parts.append(f"O{fmt(o * (1 - f))}")
    if f > 0:
        parts.append(f"F{fmt(o * f)}")
    return "".join(parts)


def item_a() -> pd.DataFrame:
    rec = pd.read_csv(OUT / "09_records_907.csv", encoding="utf-8-sig")
    rec["tier"] = rec["trace_status"].str[:2].str.strip()
    dong = rec[rec["DOI"].astype(str).str.contains(DONG, case=False) & rec["tier"].isin(["A", "B", "C", "G"])].copy()
    x_of = {"Li2Mn1O2.9795": 0.0, "Li2Mn1O2.939F0.015": 0.015, "Li2Mn1O2.9145F0.03": 0.03, "Li2Mn1O2.903F0.045": 0.045}
    dong["x"] = dong["canonical_formula"].map(x_of)
    dong = dong.sort_values(["current_rate", "cycle_value", "x", "Value"])
    dong[["canonical_formula", "x", "Value", "temperature", "current_rate", "cycle_value", "tier", "row_id"]].to_csv(
        OUT / "12a_dong_records.csv", index=False, encoding="utf-8-sig")
    # which training compositions sit near Li2MnO3 and contain F or an O deficit
    assign = pd.read_csv(SAVED / "results" / "known_control_prototype_assignment.csv")
    near = assign[assign["proto_name"].eq("Li2MnO3")][["canonical_formula", "n_records"]]
    near.to_csv(OUT / "12a_li2mno3_template_compositions.csv", index=False, encoding="utf-8-sig")
    return dong


def item_b(evaluator, models) -> pd.DataFrame:
    rows = []
    for host, li, tms, o, sub in HOSTS:
        jobs = [("F", f, formula(li, tms, o, f)) for f in F_GRID]
        for c in CR_GRID:
            t = dict(tms)
            t[sub] = t[sub] - c
            t["Cr"] = c
            jobs.append(("Cr", c, formula(li, t, o)))
        formulas = [j[2] for j in jobs]
        X = features(evaluator, formulas, 0.5, 1.0)
        full = np.asarray(evaluator.model.predict(X, thread_count=1), dtype=float)
        boot = np.column_stack([np.asarray(m.predict(X, thread_count=1), dtype=float) for m in models])
        for kind in ["F", "Cr"]:
            idx = [i for i, j in enumerate(jobs) if j[0] == kind]
            base = idx[0]
            for i in idx:
                d = boot[i] - boot[base]
                rows.append({"host": host, "kind": kind, "level": jobs[i][1], "Formula": formulas[i],
                             "pred_full": full[i], "boot_median": float(np.median(boot[i])),
                             "boot_P05": float(np.percentile(boot[i], 5)), "boot_P95": float(np.percentile(boot[i], 95)),
                             "delta_full": full[i] - full[base], "delta_boot_median": float(np.median(d)),
                             "delta_boot_P05": float(np.percentile(d, 5)), "delta_boot_P95": float(np.percentile(d, 95)),
                             "delta_n_positive": int((d > 0).sum())})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "12b_F_Cr_effects.csv", index=False, encoding="utf-8-sig")
    plot(table)
    return table


def plot(table: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "axes.edgecolor": "#52514e"})
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    for ax, kind, xlabel, scale in [(axes[0], "F", "F replacing O (% of anion sites)", 100),
                                    (axes[1], "Cr", "Cr per formula unit (replacing Mn; Co in LiCoO$_2$)", 1)]:
        for (host, *_), color in zip(HOSTS, COLORS):
            part = table[(table["host"] == host) & (table["kind"] == kind)]
            x = part["level"].to_numpy() * scale
            ax.fill_between(x, part["delta_boot_P05"], part["delta_boot_P95"], color=color, alpha=0.13, linewidth=0)
            ax.plot(x, part["delta_full"], color=color, linewidth=2, marker="o", markersize=4, label=host)
        ax.axhline(0, color="#52514e", linewidth=0.8)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", color="#e6e5e1", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(direction="out", length=3)
    axes[0].set_ylabel("Predicted change vs host (mAh g$^{-1}$)")
    axes[0].set_title("F effect", loc="left", fontsize=10, fontweight="bold")
    axes[1].set_title("Cr effect", loc="left", fontsize=10, fontweight="bold")
    axes[0].legend(loc="lower left", frameon=False, fontsize=8)
    fig.text(0.01, 0.01, "Line: full model. Band: P05–P95 of the paired difference over 100 bootstrap models. "
             "298.15 K, 0.5 C, cycle 1.", fontsize=7.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG_DIR / "12_F_Cr_effects.png", dpi=200, facecolor="#fcfcfb")
    plt.close(fig)


def item_c(evaluator) -> dict[str, object]:
    samples = load_samples()
    sel = samples[samples["paper"].isin(["Kucuk", "Lee"])]
    clean = pd.read_csv(SAVED / "data" / "clean_rows.csv")
    key_to_formula = clean.groupby("canonical_key")["canonical_formula"].first()
    n_records = clean.groupby("canonical_key").size()
    train_keys = list(evaluator.unique_train.index)
    rows = []
    for r in sel.itertuples(index=False):
        amounts = Composition(r.Formula).fractional_composition.get_el_amt_dict()
        vec = np.array([[amounts.get(e, 0.0) for e in ssb.PERIODIC_ELEMENTS]])
        dist, idx = evaluator.comp_nn.kneighbors(vec, n_neighbors=5)
        for rank, (d, j) in enumerate(zip(dist[0], idx[0]), 1):
            k = train_keys[j]
            rows.append({"paper": r.paper, "sample": r.sample, "Formula": r.Formula, "rank": rank,
                         "neighbour": key_to_formula.get(k, k), "distance": float(d),
                         "neighbour_records": int(n_records.get(k, 0))})
    nn = pd.DataFrame(rows)
    nn.to_csv(OUT / "12c_nearest_training.csv", index=False, encoding="utf-8-sig")
    rec = pd.read_csv(OUT / "09_records_907.csv", encoding="utf-8-sig")
    lnmo = rec[rec["canonical_formula"].eq("Li1.2Mn0.6Ni0.2O2")][
        ["row_id", "formula_raw", "Value", "temperature", "current_rate", "cycle_value", "trace_status", "DOI", "n_DOI"]]
    lnmo.to_csv(OUT / "12c_lnmo_records.csv", index=False, encoding="utf-8-sig")
    return {"n_lnmo_records": int(len(lnmo))}


def main() -> None:
    evaluator = ssb.CandidateEvaluator()
    models, feature_columns = load_models()
    if feature_columns != list(evaluator.feature_columns):
        raise RuntimeError("Bootstrap model feature order differs from the evaluator")
    log("a")
    item_a()
    log("b")
    item_b(evaluator, models)
    log("c")
    write_json("12_summary.json", item_c(evaluator))


if __name__ == "__main__":
    main()
