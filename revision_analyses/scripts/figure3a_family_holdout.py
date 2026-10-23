"""Figure 3A redrawn with the family-holdout value of analysis 27.

Uses the drawing code and style of make_figures.figure3 panel A (setup_style, style_axis, panel, BLUE, the
error-bar markers and value labels) from paper_release_v1/src.
Family holdout: filled marker = value after removing the two related training compositions
(outputs/27_family_holdout_summary.json, run 27_main); open gray marker = release value
(paper_release_v1/results/validation_hierarchy.csv).
Writes outputs/figure_3a/Figure_3A_revised.png and outputs/figure_3a/Figure3A_validation_hierarchy_revised.csv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
REL = BASE.parent / "paper_release_v1"
SRC = REL / "src"
OUT = BASE / "outputs"
FIG_DIR = OUT / "figure_3a"
sys.path.insert(0, str(SRC))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import make_figures as mf  # noqa: E402


def main() -> None:
    mf.setup_style()
    h = pd.read_csv(REL / "results" / "validation_hierarchy.csv")
    fam = json.loads((OUT / "27_family_holdout_summary.json").read_text(encoding="utf-8"))["runs"]["27_main"]
    rows = h.iloc[:3].copy()
    corrected = {"validation_level": "leave-one-structural-family-out (two related training compositions removed)",
                 "r2": fam["overall"]["r2"], "r2_ci95_low": fam["overall_ci95"]["r2"][0],
                 "r2_ci95_high": fam["overall_ci95"]["r2"][1], "n_records": fam["n_records"], "n_units": 112}
    original = h.iloc[3]
    table = pd.concat([rows, pd.DataFrame([corrected]), h.iloc[[3]]], ignore_index=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(FIG_DIR / "Figure3A_validation_hierarchy_revised.csv", index=False)

    labels = ["Record CV", "Exact comp.", "Fixed condition", "Family holdout"]
    values = np.r_[rows["r2"].to_numpy(float), corrected["r2"]]
    low = np.r_[rows["r2_ci95_low"].to_numpy(float), corrected["r2_ci95_low"]]
    high = np.r_[rows["r2_ci95_high"].to_numpy(float), corrected["r2_ci95_high"]]
    fig, ax = plt.subplots(figsize=(2.35, 2.65))
    fig.subplots_adjust(left=0.22, right=0.98, top=0.90, bottom=0.30)
    x = np.arange(4)
    ax.plot(x, values, color="#B8B8B8", lw=0.7, zorder=1)
    # original family-holdout value: open gray marker, slightly offset
    xo = 3.22
    ax.errorbar(xo, original["r2"], yerr=[[original["r2"] - original["r2_ci95_low"]], [original["r2_ci95_high"] - original["r2"]]],
                fmt="o", color="#9A9A9A", markerfacecolor="white", markeredgewidth=0.7, markersize=3.4,
                capsize=1.6, linewidth=0.6, zorder=2)
    ax.text(xo + 0.10, original["r2"], f"{original['r2']:.3f}", ha="left", va="center", fontsize=5.1, color="#7A7A7A")
    for idx, value in enumerate(values):
        filled = idx == 3
        ax.errorbar(idx, value, yerr=[[value - low[idx]], [high[idx] - value]], fmt="o", color=mf.BLUE,
                    markerfacecolor=mf.BLUE if filled else "white", markeredgewidth=0.7, markersize=4.0,
                    capsize=2.0, linewidth=0.7, zorder=3)
        label = f"{value:.3f}".replace("-", "−")
        if filled:
            ax.text(idx - 0.12, value, label, ha="right", va="center", fontsize=5.1)
        else:
            ax.text(idx + 0.12, value + 0.02, label, ha="left", va="bottom", fontsize=5.1)
    ax.axhline(0, color="#999999", lw=0.55, ls="--")
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("$R^2$")
    ax.set_xlim(-0.4, 4.0)
    ax.set_ylim(min(-0.10, float(low.min())) - 0.05, max(0.82, float(high.max()) + 0.04))
    mf.style_axis(ax)
    mf.panel(ax, "A")
    fig.savefig(FIG_DIR / "Figure_3A_revised.png", dpi=600, facecolor="white")
    print(table[["validation_level", "r2", "r2_ci95_low", "r2_ci95_high"]].round(3).to_string())


if __name__ == "__main__":
    main()
