"""22: ranking under test conditions (analysis 22; see revision_analyses/README.md).

Saved CatBoost full model only.  Compositions: the Li = 2 member Li2Mn0.7Cr0.3O2.865F0.135, the candidate with
the most TopK seeds (Li2.1Mn0.7Cr0.3O2.865F0.135), and each one's counterparts (a) no F, (b) no Cr,
(c) no F and no Cr (composition_counterparts.counterparts).
  a  C-rate 0.05-5 C, 10 log-spaced points; 298.15 K, cycle 1
  b  cycle 1-100, 10 log-spaced integers; 298.15 K, 0.5 C
Predictions are reported unclipped and clipped to the evaluator's range (the search uses the clipped value).
Outputs: outputs/22_*, figure outputs/figures/22_context_curves.png.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from pymatgen.core import Composition

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from composition_counterparts import counterparts  # noqa: E402
from candidate_checks import OUT, SAVED, log, ssb, write_json  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "outputs" / "figures" / "22_context_curves.png"
LI2 = "Li2Mn0.7Cr0.3O2.865F0.135"
RATES = np.round(np.logspace(math.log10(0.05), math.log10(5), 10), 4)
CYCLES = np.unique(np.round(np.logspace(0, 2, 10)).astype(int))


def predict(ev: ssb.CandidateEvaluator, formulas: list[str], rate: float, cycle: float) -> np.ndarray:
    rows = []
    for f in formulas:
        comp = Composition(f)
        rows.append(ev._composition_row({"comp": comp.get_el_amt_dict(), "Formula": f,
                                         "canonical_key": ssb.canonical_key(comp)}))
    X = pd.DataFrame(rows).reindex(columns=ev.comp_columns)
    X["temperature_K"] = ssb.FIXED_TEMPERATURE_K
    X["log1p_current_rate_C"] = math.log1p(rate)
    X["log1p_cycle_number"] = math.log1p(cycle)
    X["log1p_current_rate_x_cycle"] = math.log1p(rate * cycle)
    return np.asarray(ev.model.predict(X.reindex(columns=ev.feature_columns), thread_count=1), dtype=float)


def main() -> None:
    ev = ssb.CandidateEvaluator()
    cons = pd.read_csv(SAVED / "results" / "consensus_candidates.csv")
    top = cons.sort_values("topk_seed_count", ascending=False).iloc[0]
    groups = {}
    for focus in [LI2, top["Formula"]]:
        groups[focus] = [("candidate", focus)] + counterparts(focus)
    log(f"top-seed candidate: {top['Formula']} ({int(top['topk_seed_count'])} seeds)")
    rows = []
    for sweep, grid in [("C-rate", RATES), ("cycle", CYCLES)]:
        for value in grid:
            rate, cycle = (float(value), 1.0) if sweep == "C-rate" else (0.5, float(value))
            for focus, members in groups.items():
                pred = predict(ev, [f for _, f in members], rate, cycle)
                for (label, f), p in zip(members, pred):
                    rows.append({"sweep": sweep, "focus": focus, "member": label, "Formula": f, "C_rate": rate,
                                 "cycle": cycle, "pred_unclipped": p,
                                 "pred_clipped": float(np.clip(p, *ev.prediction_clip))})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "22_context_curves.csv", index=False, encoding="utf-8-sig")

    wide = table.pivot_table(index=["sweep", "focus", "C_rate", "cycle"], columns="member",
                             values="pred_unclipped").reset_index()
    for c in ["(a) no F", "(b) no Cr", "(c) no F, no Cr"]:
        wide[f"diff {c}"] = wide["candidate"] - wide[c]
        wide[f"above {c}"] = wide[f"diff {c}"] > 0
    wide["above all three"] = wide[[c for c in wide.columns if str(c).startswith("above (")]].all(axis=1)
    wide.to_csv(OUT / "22_context_differences.csv", index=False, encoding="utf-8-sig")
    write_json("22_summary.json", {
        "top_seed_candidate": top["Formula"], "top_seed_count": int(top["topk_seed_count"]),
        "rates": RATES.tolist(), "cycles": CYCLES.tolist(), "prediction_clip": ev.prediction_clip,
        "above_all_three_counts": wide.groupby(["sweep", "focus"])["above all three"].agg(["sum", "size"])
        .reset_index().to_dict(orient="records")})

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey="row")
    styles = {"candidate": ("k", "-", "o"), "(a) no F": ("tab:blue", "--", "s"),
              "(b) no Cr": ("tab:orange", "--", "^"), "(c) no F, no Cr": ("tab:green", ":", "v")}
    for col, focus in enumerate(groups):
        for row, (sweep, xcol) in enumerate([("C-rate", "C_rate"), ("cycle", "cycle")]):
            ax = axes[row, col]
            part = table[table["sweep"].eq(sweep) & table["focus"].eq(focus)]
            for label, (color, ls, marker) in styles.items():
                sub = part[part["member"].eq(label)]
                ax.plot(sub[xcol], sub["pred_unclipped"], color=color, ls=ls, marker=marker, ms=4,
                        label=f"{label}: {sub['Formula'].iloc[0]}")
            ax.set_xscale("log")
            ax.set_xlabel("C-rate (C), 298.15 K, cycle 1" if sweep == "C-rate" else "cycle, 298.15 K, 0.5 C")
            ax.set_ylabel("predicted capacity (mAh/g)")
            ax.set_title(focus, fontsize=10)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    log(f"figure {FIG}")


if __name__ == "__main__":
    main()
