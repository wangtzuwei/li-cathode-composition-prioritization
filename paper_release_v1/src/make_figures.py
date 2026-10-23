#!/usr/bin/env python3
"""Generate the five manuscript figures from the canonical release tables."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import sys
from pathlib import Path

import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from PIL import Image
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "paper_release_v1"
DATA = RELEASE / "data"
RESULTS = RELEASE / "results"
FIGURES = RELEASE / "figures"
SOURCES = FIGURES / "source_data"
FIGURES.mkdir(parents=True, exist_ok=True)
SOURCES.mkdir(parents=True, exist_ok=True)

# The visual system below is intentionally inherited from the author's
# existing Figures 2-5: compact Arial-like type, full boxed axes, inward ticks,
# restrained grids, the established blue/orange/green/red/purple palette, and
# 600-dpi output.  New canonical data are redrawn, but the author's figure
# language is preserved so the revised panels do not require journal reformatting.
PAL = {
    "blue": "#3574A7",
    "orange": "#E8843A",
    "green": "#4FA858",
    "red": "#C44E52",
    "purple": "#8172B3",
    "gray": "#999999",
    "pink": "#DA8BC3",
    "brown": "#937860",
    "cyan": "#64B5CD",
    "olive": "#CCB974",
}
BLUE = PAL["blue"]
ORANGE = PAL["orange"]
GREEN = PAL["green"]
PURPLE = PAL["purple"]
RED = PAL["red"]
GRAY = PAL["gray"]
GOLD = PAL["olive"]
LIGHT = "#E7EDF3"

PROTO_LABEL = {
    "LiCoO2": r"LiCoO$_2$",
    "LiNiO2": r"LiNiO$_2$",
    "NCA": "NCA",
    "NMC-811": "NMC-811",
    "NMC-622": "NMC-622",
    "NMC-532": "NMC-532",
    "Li-rich-NMC": "Li-rich NMC",
    "Li2MnO3": r"Li$_2$MnO$_3$",
    "LNMO-spinel": "LNMO spinel",
    "LiMn2O4": r"LiMn$_2$O$_4$",
    "LiFePO4": r"LiFePO$_4$",
    "LiMnPO4": r"LiMnPO$_4$",
}

FAMILY = {
    "LiCoO2": "Layered", "LiNiO2": "Layered", "NCA": "Layered",
    "NMC-811": "Layered", "NMC-622": "Layered", "NMC-532": "Layered",
    "Li-rich-NMC": "Layered", "Li2MnO3": "Layered",
    "LNMO-spinel": "Spinel", "LiMn2O4": "Spinel",
    "LiFePO4": "Polyanion", "LiMnPO4": "Polyanion",
}
FAMILY_COLOR = {"Layered": BLUE, "Spinel": ORANGE, "Polyanion": GREEN}
FAMILY_LIGHT = {"Layered": "#C9DCEF", "Spinel": "#F6D1B3", "Polyanion": "#CBE6CD"}

ARM_ORDER = [
    "nsga2__floor_off",
    "nsga2__floor_on",
    "random__floor_off",
    "random__floor_on",
]
ARM_LABEL = {
    "nsga2__floor_off": "NSGA-II\nfloor off",
    "nsga2__floor_on": "NSGA-II\nfloor on",
    "random__floor_off": "Random\nfloor off",
    "random__floor_on": "Random\nfloor on",
}
ARM_COLOR = {
    "nsga2__floor_off": PURPLE,
    "nsga2__floor_on": RED,
    "random__floor_off": BLUE,
    "random__floor_on": GREEN,
}


def setup_style() -> None:
    sns.set_theme(style="white")
    matplotlib.rcParams.update(
        {
            "text.usetex": False,
            "mathtext.fontset": "dejavusans",
            "svg.fonttype": "none",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica Neue", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.45,
            "ytick.major.width": 0.45,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "lines.linewidth": 1.0,
            "savefig.dpi": 600,
            "figure.dpi": 150,
            "savefig.bbox": None,
        }
    )


def style_axis(ax: plt.Axes, grid: bool = True) -> None:
    for spine in ax.spines.values():
        spine.set_color("#555555")
        spine.set_linewidth(0.5)
    ax.tick_params(length=2.0, width=0.4)
    if grid:
        ax.grid(True, alpha=0.08, linewidth=0.2, color="#999999")
    else:
        ax.grid(False)


def panel(ax: plt.Axes, label: str) -> None:
    ax.text(-0.14, 1.13, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def save(fig: plt.Figure, stem: str) -> None:
    # The author's submitted Figures 2-5 use an exact 7.35-inch canvas rather
    # than a tight-cropped canvas.  Retaining the full canvas preserves their
    # panel spacing and produces the same 4410-pixel width at 600 dpi.
    png_target = FIGURES / f"{stem}.png"
    png_staging = FIGURES / f".{stem}.staging.png"
    pdf_target = FIGURES / f"{stem}.pdf"
    pdf_staging = FIGURES / f".{stem}.staging.pdf"
    fig.savefig(png_staging, dpi=600, facecolor="white", format="png")
    # Force a complete decode before replacing the release asset.  This makes
    # a short filesystem write fail loudly instead of silently leaving a
    # truncated PNG that Word can embed but readers cannot reliably render.
    with Image.open(png_staging) as image:
        image.load()
    os.replace(png_staging, png_target)
    fig.savefig(pdf_staging, facecolor="white", format="pdf")
    os.replace(pdf_staging, pdf_target)
    plt.close(fig)


def figure1() -> None:
    # Reuse the author's already-drawn conceptual Figure 1 byte-for-byte.
    source = FIGURES / "source_assets" / "Figure1_original_concept.png"
    target = FIGURES / "Figure1_revised_workflow.png"
    if not source.exists():
        raise FileNotFoundError(f"Missing retained Figure 1 source asset: {source}")
    shutil.copy2(source, target)

    image = mpimg.imread(source)
    height = 7.35 * image.shape[0] / image.shape[1]
    fig = plt.figure(figsize=(7.35, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image)
    ax.axis("off")
    fig.savefig(FIGURES / "Figure1_revised_workflow.pdf", bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close(fig)


def figure2() -> None:
    clean = pd.read_csv(DATA / "clean_rows.csv")
    groups = pd.read_csv(DATA / "formula_group_summary.csv")
    assignment = pd.read_csv(RESULTS / "known_control_prototype_assignment.csv")[["canonical_key", "proto_name"]]
    mapped = clean.merge(assignment, on="canonical_key", how="left")
    assigned = mapped[mapped["proto_name"].notna()].copy()
    group_medians = (
        assigned.groupby(["proto_name", "canonical_key"], as_index=False)["Value"].median()
    )
    groups[["canonical_key", "n_records", "n_raw_strings", "raw_strings"]].to_csv(
        SOURCES / "Figure2A_canonical_multiplicity.csv", index=False
    )
    clean[["row_id", "canonical_key", "temperature", "current_rate", "cycle_value"]].to_csv(
        SOURCES / "Figure2B_measurement_context.csv", index=False
    )
    assigned[["row_id", "canonical_key", "proto_name", "Value"]].to_csv(
        SOURCES / "Figure2C_mapped_capacity.csv", index=False
    )

    # Preserve the author's compact 1 x 3 Figure-2 geometry.
    fig = plt.figure(figsize=(7.35, 3.72))
    outer = fig.add_gridspec(
        1, 3, width_ratios=[0.86, 0.92, 1.46],
        left=0.075, right=0.980, top=0.91, bottom=0.16, wspace=0.48,
    )
    ax_a = fig.add_subplot(outer[0, 0])
    bgrid = outer[0, 1].subgridspec(3, 1, hspace=0.48)
    axes_b = [fig.add_subplot(bgrid[idx, 0]) for idx in range(3)]
    ax_c = fig.add_subplot(outer[0, 2])

    counts = groups["n_records"].to_numpy(int)
    edges = np.array([0.5, 1.5, 2.5, 4.5, 8.5, 16.5, 32.5, 64.5, 128.5])
    hist, _ = np.histogram(counts, bins=edges)
    widths = np.diff(edges)
    ax_a.bar(edges[:-1], hist, width=widths, align="edge", color=BLUE, alpha=0.75, edgecolor="white", linewidth=0.3)
    ax_a.axvline(np.median(counts), color=ORANGE, ls="--", lw=0.9)
    ax_a.set_xscale("log", base=2)
    ax_a.set_xticks([1, 2, 4, 8, 16, 32, 64, 128], ["1", "2", "4", "8", "16", "32", "64", "128"])
    ax_a.set_xlabel("Records per composition")
    ax_a.set_ylabel("Number of canonical compositions")
    ax_a.tick_params(axis="x", labelsize=6.0)
    style_axis(ax_a)

    context_specs = [
        ("temperature", "Temperature (K)", False, 18),
        ("current_rate", "C-rate", True, np.logspace(-2, 2, 18)),
        ("cycle_value", "Cycle number", True, np.logspace(0, 3.2, 18)),
    ]
    for idx, (column, xlabel, log_x, bins) in enumerate(context_specs):
        values = pd.to_numeric(clean[column], errors="coerce").dropna().to_numpy(float)
        if log_x:
            values = values[values > 0]
        axes_b[idx].hist(values, bins=bins, color=BLUE, alpha=0.72, edgecolor="white", linewidth=0.25)
        axes_b[idx].axvline(np.median(values), color=ORANGE, ls="--", lw=0.8)
        if log_x:
            axes_b[idx].set_xscale("log")
        axes_b[idx].set_xlabel(xlabel, labelpad=1)
        axes_b[idx].set_ylabel("Records" if idx == 1 else "")
        style_axis(axes_b[idx])
    context_legend = [
        Line2D([0], [0], color=BLUE, lw=5, alpha=0.72, label="Records"),
        Line2D([0], [0], color=ORANGE, lw=0.9, ls="--", label="Median"),
    ]
    axes_b[0].legend(
        handles=context_legend, frameon=False, ncol=2, loc="upper right",
        fontsize=5.2, handlelength=1.2, columnspacing=0.7,
    )

    order = [
        "NCA", "NMC-811", "NMC-622", "NMC-532", "LiNiO2", "LiCoO2",
        "Li-rich-NMC", "Li2MnO3", "LNMO-spinel", "LiMn2O4", "LiFePO4", "LiMnPO4",
    ]
    palette = {proto: FAMILY_LIGHT[FAMILY[proto]] for proto in order}
    sns.boxplot(
        data=assigned, x="Value", y="proto_name", order=order, palette=palette, hue="proto_name",
        legend=False, ax=ax_c, showfliers=False, linewidth=0.55, saturation=0.75,
        boxprops={"edgecolor": "#777777"}, whiskerprops={"color": "#777777", "linewidth": 0.5},
        capprops={"color": "#777777", "linewidth": 0.5}, medianprops={"color": "#555555", "linewidth": 0.65},
    )
    rng = np.random.default_rng(42)
    for y_pos, proto in enumerate(order):
        record_values = assigned.loc[assigned["proto_name"] == proto, "Value"].to_numpy(float)
        if len(record_values):
            ax_c.scatter(
                record_values, y_pos + rng.uniform(-0.19, 0.19, len(record_values)),
                s=4.5, color=FAMILY_COLOR[FAMILY[proto]], alpha=0.27, edgecolors="none", rasterized=True, zorder=2,
            )
        balanced_values = group_medians.loc[group_medians["proto_name"] == proto, "Value"].to_numpy(float)
        if len(balanced_values):
            ax_c.scatter(
                np.median(balanced_values), y_pos, marker="D", s=24,
                facecolor="white", edgecolor="#333333", linewidth=0.55, zorder=5,
            )
    ax_c.set_yticks(range(len(order)), [PROTO_LABEL[item] for item in order])
    ax_c.set_xlabel("Measured capacity (mAh g$^{-1}$)")
    ax_c.set_ylabel("")
    ax_c.set_xlim(0, 350)
    legend_handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=FAMILY_LIGHT[name], markeredgecolor="none", label=name, markersize=5)
        for name in ["Layered", "Spinel", "Polyanion"]
    ] + [
        Line2D([0], [0], marker="D", color="none", markerfacecolor="white", markeredgecolor="#333333", label="Formula median", markersize=4)
    ]
    ax_c.legend(
        handles=legend_handles, frameon=False, fontsize=4.8, ncol=4,
        loc="upper left", handletextpad=0.25, columnspacing=0.55,
    )
    for x_position, label in [(0.035, "A"), (0.325, "B"), (0.595, "C")]:
        fig.text(x_position, 0.97, label, va="top", ha="left", fontsize=11, fontweight="bold")
    style_axis(ax_c)
    save(fig, "Figure2_revised_dataset")


def parity_axis(ax: plt.Axes, frame: pd.DataFrame, annotation: str) -> None:
    x = frame["y_true"].to_numpy(float)
    y = frame["y_pred"].to_numpy(float)
    # Match the submitted figure: colour encodes local plotting density, not
    # residual magnitude.  Dense points are drawn last so the centre of the
    # distribution remains visible.
    density = gaussian_kde(np.vstack([x, y]))(np.vstack([x, y]))
    order = np.argsort(density)
    norm = Normalize(vmin=float(np.percentile(density, 2)), vmax=float(np.percentile(density, 98)))
    ax.scatter(
        x[order], y[order], s=5.5, c=plt.cm.RdYlGn_r(norm(density[order])),
        alpha=0.70, edgecolors="none", rasterized=True, zorder=2,
    )
    hi = float(max(x.max(), y.max()) * 1.04)
    ax.plot([0, hi], [0, hi], ls="--", color="#999999", lw=0.55, zorder=1)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Measured capacity (mAh g$^{-1}$)")
    ax.set_ylabel("Predicted capacity (mAh g$^{-1}$)")
    ax.text(
        0.04, 0.96, annotation, transform=ax.transAxes, va="top", ha="left", fontsize=5.2,
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#CCCCCC", "linewidth": 0.3, "alpha": 0.92},
    )
    style_axis(ax)


def figure3() -> None:
    results = json.loads((RESULTS / "results_for_manuscript.json").read_text(encoding="utf-8"))
    all_oof = pd.read_csv(RESULTS / "oof_canonical_composition_heldout_all_models.csv")
    held = all_oof[all_oof["model"] == "catboost__composition_plus_context"].copy()
    distance = pd.read_csv(RESULTS / "distance_error_formula_level.csv")
    hierarchy = pd.read_csv(RESULTS / "validation_hierarchy.csv")
    hierarchy.to_csv(SOURCES / "Figure3A_validation_hierarchy.csv", index=False)
    held.to_csv(SOURCES / "Figure3B_canonical_heldout.csv", index=False)
    distance.to_csv(SOURCES / "Figure3C_distance_error_formula_level.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.78))
    fig.subplots_adjust(left=0.075, right=0.980, top=0.89, bottom=0.23, wspace=0.45)
    r = results["record_cv_catboost_context"]
    h = results["canonical_composition_heldout_catboost_context"]
    d = results["distance_error_formula_level"]
    nominal = json.loads((RESULTS / "nominal_context_validation.json").read_text(encoding="utf-8"))
    family = json.loads((RESULTS / "leave_one_family_out_validation.json").read_text(encoding="utf-8"))
    power = json.loads((RESULTS / "distance_error_power.json").read_text(encoding="utf-8"))
    inference = pd.read_csv(RESULTS / "primary_inference_family.csv").set_index("test")
    distance_holm = float(inference.loc["distance_error_association", "holm_adjusted_p"])

    labels = ["Record CV", "Exact comp.", "Nominal slice", "Leave family"]
    values = hierarchy["r2"].to_numpy(float)
    low = hierarchy["r2_ci95_low"].to_numpy(float)
    high = hierarchy["r2_ci95_high"].to_numpy(float)
    x_positions = np.arange(4)
    axes[0].plot(x_positions, values, color="#B8B8B8", lw=0.7, zorder=1)
    for idx, value in enumerate(values):
        if np.isfinite(low[idx]) and np.isfinite(high[idx]):
            axes[0].errorbar(
                idx, value, yerr=[[value - low[idx]], [high[idx] - value]],
                fmt="o", color=BLUE, markerfacecolor="white", markeredgewidth=0.7,
                markersize=4.0, capsize=2.0, linewidth=0.7, zorder=3,
            )
        else:
            axes[0].scatter(idx, value, s=18, color=BLUE, edgecolor="white", linewidth=0.5, zorder=3)
        axes[0].text(idx, value + 0.045, f"{value:.3f}", ha="center", va="bottom", fontsize=5.1)
    axes[0].axhline(0, color="#999999", lw=0.55, ls="--")
    axes[0].set_xticks(x_positions, labels, rotation=25, ha="right")
    axes[0].set_ylabel("$R^2$")
    axes[0].set_ylim(min(-0.10, float(np.nanmin(low)) - 0.05), max(0.82, float(np.nanmax(high)) + 0.04))
    style_axis(axes[0])
    panel(axes[0], "A")
    parity_axis(
        axes[1], held,
        f"Leave-one-composition-out OOF\n$R^2$ = {h['r2']:.3f}; MAE = {h['mae']:.1f} mAh g$^{{-1}}$\nn = 907; 171 compositions",
    )
    x_dist = distance["median_normalized_5nn_distance"].to_numpy(float)
    y_err = distance["median_abs_error"].to_numpy(float)
    axes[2].scatter(x_dist, y_err, s=7, color="#B8B8B8", alpha=0.38, edgecolors="none", rasterized=True)
    bins = pd.qcut(pd.Series(x_dist), q=8, duplicates="drop")
    binned = pd.DataFrame({"x": x_dist, "y": y_err, "bin": bins}).groupby("bin", observed=True).agg(
        x=("x", "median"), y=("y", "median"), lo=("y", lambda values: np.quantile(values, 0.25)), hi=("y", lambda values: np.quantile(values, 0.75))
    )
    axes[2].errorbar(
        binned["x"], binned["y"],
        yerr=np.vstack([binned["y"] - binned["lo"], binned["hi"] - binned["y"]]),
        fmt="o", color=BLUE, markerfacecolor=BLUE, markeredgecolor="white", markeredgewidth=0.4,
        markersize=3.3, capsize=2, linewidth=0.65, zorder=4,
    )
    axes[2].axvline(1.0, color=ORANGE, ls="--", lw=0.8, label="training-fold $d_{95}$")
    axes[2].set_ylabel("Median absolute OOF error (mAh g$^{-1}$)")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Median normalized 5NN distance")
    axes[2].text(
        0.04, 0.96,
        f"Formula-level distance-error inference\nSpearman $\\rho$ = {d['spearman_rho']:.3f}\n95% CI [{d['rho_bootstrap_ci95'][0]:.3f}, {d['rho_bootstrap_ci95'][1]:.3f}]\nraw p = {d['formula_label_permutation_p_two_sided']:.3f}; Holm p = {distance_holm:.3f}\napprox. power at $\\rho$ = 0.13: {100*power['fisher_z_approximate_power_at_observed_rho']:.0f}%",
        transform=axes[2].transAxes, va="top", fontsize=5.1,
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#CCCCCC", "linewidth": 0.3, "alpha": 0.92},
    )
    axes[2].legend(frameon=False, loc="lower right", fontsize=5.5)
    style_axis(axes[2])
    panel(axes[1], "B")
    panel(axes[2], "C")
    save(fig, "Figure3_revised_model_validation")


def dot_summary(ax: plt.Axes, frame: pd.DataFrame, metric: str, ylabel: str) -> None:
    rng = np.random.default_rng(7)
    for idx, arm in enumerate(ARM_ORDER):
        values = frame.loc[frame["arm"] == arm, metric].to_numpy(float)
        jitter = rng.uniform(-0.07, 0.07, len(values))
        ax.scatter(np.full(len(values), idx) + jitter, values, color=ARM_COLOR[arm], s=13, alpha=0.78, zorder=3)
        ax.errorbar(idx, values.mean(), yerr=values.std(ddof=1), fmt="D", color="#222222", mfc="white", ms=3.5, capsize=2.5, lw=0.7, zorder=4)
    ax.set_xticks(range(len(ARM_ORDER)), [ARM_LABEL[item] for item in ARM_ORDER])
    ax.tick_params(axis="x", labelsize=5.7)
    ax.set_ylabel(ylabel)
    style_axis(ax)


def figure4() -> None:
    summary = pd.read_csv(RESULTS / "search_seed_summary.csv")
    history = pd.read_csv(RESULTS / "nsga_generation_history.csv")
    paired = pd.read_csv(RESULTS / "search_paired_seed_comparisons.csv")
    summary.to_csv(SOURCES / "Figure4_seed_summary.csv", index=False)
    history.to_csv(SOURCES / "Figure4_nsga_history.csv", index=False)

    # Retain the submitted Figure-4 rhythm: trajectory, central comparison,
    # and prototype-count bars.  Only the evidence inside those panels changes
    # to show the matched-seed and prototype-floor comparisons.
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.78))
    fig.subplots_adjust(left=0.090, right=0.985, top=0.89, bottom=0.20, wspace=0.62)
    ax = axes[0]
    for seed, part in history.groupby("seed"):
        ax.plot(part["generation"], part["top50_median_y_pred"], color=BLUE, alpha=0.18, lw=0.45)
        ax.plot(part["generation"], part["top50_median_y_discounted"], color=ORANGE, alpha=0.14, lw=0.45, ls="--")
    agg = history.groupby("generation").agg(
        raw_mean=("top50_median_y_pred", "mean"), raw_sd=("top50_median_y_pred", "std"),
        discounted_mean=("top50_median_y_discounted", "mean"), discounted_sd=("top50_median_y_discounted", "std"),
    ).reset_index()
    x = agg["generation"].to_numpy(float)
    for mean_col, sd_col, color, label in [
        ("raw_mean", "raw_sd", BLUE, "Raw prediction"),
        ("discounted_mean", "discounted_sd", ORANGE, "Distance-discounted prediction"),
    ]:
        mean = agg[mean_col].to_numpy(float)
        sd = agg[sd_col].fillna(0).to_numpy(float)
        line_style = "-" if color == BLUE else "--"
        ax.plot(x, mean, color=color, lw=1.1, ls=line_style, label=label)
        ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.10)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Top-50 median capacity\n(mAh g$^{-1}$)", labelpad=1.5)
    ax.legend(frameon=False, fontsize=5.1, loc="lower right")
    panel(ax, "A")
    style_axis(ax)

    ax = axes[1]
    dot_summary(ax, summary, "feasible_hypervolume", "Post-screen feasible hypervolume")
    proposal_hv = paired[
        (paired["comparison"] == "proposal_effect_floor_on")
        & (paired["metric"] == "feasible_hypervolume")
    ].iloc[0]
    ax.text(
        0.04, 0.96,
        "Primary: post-screen feasible HV\n"
        f"NSGA-II − random = {proposal_hv['mean_paired_delta']:.2f}\n"
        f"95% CI [{proposal_hv['paired_bootstrap_ci95_low']:.2f}, {proposal_hv['paired_bootstrap_ci95_high']:.2f}]\n"
        f"positive in {int(proposal_hv['n_positive'])}/{int(proposal_hv['n_seeds'])} seeds",
        transform=ax.transAxes, ha="left", va="top", fontsize=4.5,
        bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "#CCCCCC", "linewidth": 0.3, "alpha": 0.92},
    )
    panel(ax, "B")

    ax = axes[2]
    proto_order = [
        "LiCoO2", "LiNiO2", "NCA", "NMC-811", "NMC-622", "NMC-532",
        "Li-rich-NMC", "Li2MnO3", "LNMO-spinel", "LiMn2O4", "LiFePO4", "LiMnPO4",
    ]
    count_rows: list[dict[str, object]] = []
    for arm in ["nsga2__floor_on", "random__floor_on"]:
        for seed in sorted(summary.loc[summary["arm"].eq(arm), "seed"].astype(int).unique()):
            topk = pd.read_csv(RESULTS / "search_candidates" / f"{arm}_seed{seed}_topk_screened.csv")
            counts = topk["proto_name"].value_counts()
            for proto in proto_order:
                count_rows.append({"arm": arm, "seed": seed, "proto_name": proto, "count": int(counts.get(proto, 0))})
    count_frame = pd.DataFrame(count_rows)
    count_frame.to_csv(SOURCES / "Figure4C_floor_on_prototype_counts.csv", index=False)
    count_summary = count_frame.groupby(["arm", "proto_name"])["count"].agg(["mean", "std"]).reset_index()
    y = np.arange(len(proto_order))
    for arm, offset, color, label in [
        ("nsga2__floor_on", -0.17, BLUE, "NSGA-II"),
        ("random__floor_on", 0.17, "#BFBFBF", "Random"),
    ]:
        part = count_summary[count_summary["arm"] == arm].set_index("proto_name").reindex(proto_order)
        values = part["mean"].to_numpy(float)
        errors = part["std"].fillna(0).to_numpy(float)
        ax.barh(y + offset, values, height=0.30, xerr=errors, color=color, alpha=0.90, label=label,
                error_kw={"elinewidth": 0.45, "capsize": 1.2, "capthick": 0.45})
    ax.axvline(10, color="#888888", ls="--", lw=0.6)
    ax.set_yticks(y, [PROTO_LABEL[item] for item in proto_order])
    ax.invert_yaxis()
    ax.set_xlabel("Candidates per prototype")
    ax.set_ylabel("Cathode prototype", fontsize=7.0, labelpad=1.0)
    ax.tick_params(axis="y", labelsize=6.0)
    ax.legend(frameon=False, fontsize=5.2, loc="lower right")
    panel(ax, "C")
    style_axis(ax)
    save(fig, "Figure4_revised_search_benchmark")


def figure5() -> None:
    summary = pd.read_csv(RESULTS / "search_seed_summary.csv")
    threshold = pd.read_csv(RESULTS / "radius_threshold_sensitivity.csv")
    li_tm_threshold = pd.read_csv(RESULTS / "li_tm_window_sensitivity.csv")
    known = pd.read_csv(RESULTS / "known_control_screened.csv")
    known_summary = json.loads((RESULTS / "known_control_summary.json").read_text(encoding="utf-8"))
    primary = json.loads((RESULTS / "primary_analysis_summary.json").read_text(encoding="utf-8"))
    reference = pd.read_csv(RESULTS / "primary_reference_topk_risk_tiers.csv")
    threshold.to_csv(SOURCES / "Figure5_threshold_sensitivity.csv", index=False)
    li_tm_threshold.to_csv(SOURCES / "Figure5_li_tm_window_sensitivity.csv", index=False)
    reference.to_csv(SOURCES / "Figure5_reference_screening.csv", index=False)
    known.to_csv(SOURCES / "Figure5_positive_controls.csv", index=False)

    # Preserve the author's exact screening-cascade format, including the two
    # stacked sensitivity axes in Panel C.
    fig = plt.figure(figsize=(7.35, 2.78))
    grid = fig.add_gridspec(
        1, 3, width_ratios=[0.98, 1.52, 1.06],
        left=0.065, right=0.980, top=0.87, bottom=0.19, wspace=0.46,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    heat_grid = grid[0, 1].subgridspec(1, 2, wspace=0.07)
    ax_b_known = fig.add_subplot(heat_grid[0, 0])
    ax_b_ref = fig.add_subplot(heat_grid[0, 1])
    sensitivity_grid = grid[0, 2].subgridspec(2, 1, hspace=0.78)
    ax_c_li = fig.add_subplot(sensitivity_grid[0, 0])
    ax_c_radius = fig.add_subplot(sensitivity_grid[1, 0])

    p90 = float(known_summary["positive_control_radius_p90_after_g1_g2"])
    p95 = float(known_summary["positive_control_radius_p95_after_g1_g2"])
    stages = ["Initial", "G1\nIonic", "G2\nLi/TM", "G3\nP95", "G4\nRedox"]
    for arm, color, marker, label in [
        ("nsga2__floor_on", BLUE, "o", "NSGA-II"),
        ("random__floor_on", "#888888", "s", "Random"),
    ]:
        per_seed = []
        for seed in sorted(summary.loc[summary["arm"] == arm, "seed"].astype(int).unique()):
            frame = pd.read_csv(RESULTS / "search_candidates" / f"{arm}_seed{seed}_topk_screened.csv")
            g1 = frame["g1_pass"].astype(bool)
            g2 = frame["g2_pass"].astype(bool)
            g4 = frame["g4_pass"].astype(bool)
            radius = pd.to_numeric(frame["radius_mismatch"], errors="coerce")
            per_seed.append(
                [
                    len(frame),
                    int(g1.sum()),
                    int((g1 & g2).sum()),
                    int((g1 & g2 & radius.le(p95)).sum()),
                    int((g1 & g2 & radius.le(p95) & g4).sum()),
                ]
            )
        matrix = np.asarray(per_seed, dtype=float) / 500.0
        mean = matrix.mean(axis=0)
        sd = matrix.std(axis=0, ddof=1)
        x = np.arange(len(stages))
        ax_a.plot(x, mean * 100, marker=marker, ms=2.5, color=color, lw=0.9, label=label)
        ax_a.fill_between(x, (mean - sd) * 100, (mean + sd) * 100, color=color, alpha=0.07)
    known_values = np.array(
        [
            len(known),
            known["pass_g1"].sum(),
            known["pass_g1_g2"].sum(),
            (known["pass_g1_g2"].astype(bool) & known["radius_mismatch"].le(p95)).sum(),
            (known["pass_g1_g2"].astype(bool) & known["radius_mismatch"].le(p95) & known["g4_pass"].astype(bool)).sum(),
        ], dtype=float,
    ) / len(known)
    ax_a.plot(
        np.arange(len(stages)), known_values * 100, marker="D", ms=2.5,
        markerfacecolor="white", markeredgecolor=GREEN, color=GREEN, lw=0.9,
        label="Positive controls",
    )
    ax_a.set_xticks(np.arange(len(stages)), stages)
    ax_a.set_ylabel("Cumulative retention (%)")
    ax_a.set_ylim(-2, 104)
    ax_a.legend(frameon=False, fontsize=5.0, loc="lower left", labelspacing=0.25, handlelength=1.4)
    panel(ax_a, "A")
    style_axis(ax_a)

    order = [
        "LiCoO2", "LiNiO2", "NCA", "NMC-811", "NMC-622", "NMC-532",
        "Li-rich-NMC", "Li2MnO3", "LNMO-spinel", "LiMn2O4", "LiFePO4", "LiMnPO4",
    ]
    outcomes = ["G1", "G2", "G3", "G4", "Pass"]

    def outcome_table(frame: pd.DataFrame) -> np.ndarray:
        g1 = frame["g1_pass"].astype(bool)
        g2 = frame["g2_pass"].astype(bool)
        g4 = frame["g4_pass"].astype(bool)
        radius = pd.to_numeric(frame["radius_mismatch"], errors="coerce")
        outcome = np.select(
            [~g1, g1 & ~g2, g1 & g2 & radius.gt(p95), g1 & g2 & radius.le(p95) & ~g4],
            ["G1", "G2", "G3", "G4"],
            default="Pass",
        )
        table = pd.crosstab(frame["proto_name"], outcome, normalize="index")
        return table.reindex(index=order, columns=outcomes, fill_value=0).to_numpy(float) * 100

    matrices = [outcome_table(known), outcome_table(reference)]
    heat_titles = ["Positive controls", "Reference TopK"]
    sample_sizes = [known["proto_name"].value_counts(), reference["proto_name"].value_counts()]
    heat_image = None
    for idx, (axis, matrix, title) in enumerate(zip([ax_b_known, ax_b_ref], matrices, heat_titles)):
        heat_image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=100, aspect="auto", interpolation="nearest")
        axis.set_xticks(range(len(outcomes)), outcomes, rotation=0)
        axis.set_yticks(range(len(order)))
        if idx == 0:
            axis.set_yticklabels(
                [f"{PROTO_LABEL[item]} (n={int(sample_sizes[idx].get(item, 0))})" for item in order]
            )
        else:
            axis.set_yticklabels([])
            for row_idx, proto in enumerate(order):
                axis.text(
                    5.05, row_idx, f"n={int(sample_sizes[idx].get(proto, 0))}",
                    ha="left", va="center", fontsize=3.8, color="#777777", clip_on=False,
                )
        axis.set_title(title, fontsize=6.2, pad=1)
        axis.tick_params(axis="both", labelsize=4.6, top=False, right=False, length=1.5)
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = matrix[row_idx, col_idx]
                if value >= 0.5:
                    axis.text(
                        col_idx, row_idx, f"{value:.0f}", ha="center", va="center",
                        fontsize=3.6, color="white" if value >= 55 else "#333333",
                    )
        for spine in axis.spines.values():
            spine.set_color("#777777")
            spine.set_linewidth(0.4)
        for boundary in [7.5, 9.5]:
            axis.axhline(boundary, color="#999999", lw=0.45)
    ax_b_known.set_ylabel("Prototype")
    cax = fig.add_axes([0.397, 0.925, 0.122, 0.014])
    colorbar = fig.colorbar(heat_image, cax=cax, orientation="horizontal")
    colorbar.set_ticks([0, 50, 100])
    colorbar.ax.tick_params(labelsize=4.0, length=1.2, pad=0.5)
    colorbar.ax.set_title("Outcome fraction (%)", fontsize=4.4, pad=1)
    panel(ax_b_known, "B")

    primary_pool = primary["primary_arm"]
    li_part = li_tm_threshold[li_tm_threshold["pool"] == primary_pool]
    li_agg = li_part.groupby("relative_li_tm_window_width")["conditional_retention"].agg(["mean", "std"]).reset_index()
    li_x = li_agg["relative_li_tm_window_width"].to_numpy(float)
    li_mean = li_agg["mean"].to_numpy(float)
    li_sd = li_agg["std"].fillna(0).to_numpy(float)
    primary_label = "NSGA-II floor on" if primary_pool == "nsga2__floor_on" else "Random floor on"
    ax_c_li.plot(li_x, li_mean * 100, color=BLUE, marker="o", ms=2.0, lw=0.9, label=primary_label)
    ax_c_li.fill_between(li_x, (li_mean - li_sd) * 100, (li_mean + li_sd) * 100, color=BLUE, alpha=0.08)
    li_control = li_tm_threshold[li_tm_threshold["pool"] == "known_positive_controls"]
    z = 1.96
    li_n = li_control["n_reaching_g2"].to_numpy(float)
    li_phat = li_control["conditional_retention"].to_numpy(float)
    li_denom = 1 + z**2 / li_n
    li_center = (li_phat + z**2 / (2 * li_n)) / li_denom
    li_half = z * np.sqrt(li_phat * (1 - li_phat) / li_n + z**2 / (4 * li_n**2)) / li_denom
    ax_c_li.fill_between(
        li_control["relative_li_tm_window_width"].to_numpy(float),
        np.maximum(0, li_center - li_half) * 100,
        np.minimum(1, li_center + li_half) * 100,
        color="#333333", alpha=0.07, linewidth=0,
    )
    ax_c_li.plot(
        li_control["relative_li_tm_window_width"], li_control["conditional_retention"] * 100,
        color="#333333", marker="s", markerfacecolor="white", ms=2.0, lw=0.8, label="Positive controls",
    )
    ax_c_li.axvline(1.0, color="#999999", ls="--", lw=0.65)
    ax_c_li.set_xlim(0.5, 1.5)
    ax_c_li.set_ylim(-2, 104)
    ax_c_li.set_xticks([0.5, 1.0, 1.5])
    ax_c_li.set_xlabel("Relative Li/TM window width", labelpad=1)
    ax_c_li.text(
        0.04, 0.72, "G2 Li/TM window", transform=ax_c_li.transAxes, fontsize=5.0,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.4},
    )
    style_axis(ax_c_li)

    part = threshold[threshold["pool"] == primary_pool]
    agg = part.groupby("radius_threshold")["conditional_retention"].agg(["mean", "std"]).reset_index()
    x = agg["radius_threshold"].to_numpy(float)
    mean = agg["mean"].to_numpy(float)
    sd = agg["std"].fillna(0).to_numpy(float)
    ax_c_radius.plot(x, mean * 100, color=BLUE, lw=1.0, marker="o", ms=2.0, markevery=3, label=primary_label)
    ax_c_radius.fill_between(x, (mean - sd) * 100, (mean + sd) * 100, color=BLUE, alpha=0.09)
    control = threshold[threshold["pool"] == "known_positive_controls"]
    control_n = control["n_reaching_g3"].to_numpy(float)
    control_phat = control["conditional_retention"].to_numpy(float)
    control_denom = 1 + z**2 / control_n
    control_center = (control_phat + z**2 / (2 * control_n)) / control_denom
    control_half = z * np.sqrt(
        control_phat * (1 - control_phat) / control_n + z**2 / (4 * control_n**2)
    ) / control_denom
    ax_c_radius.fill_between(
        control["radius_threshold"].to_numpy(float),
        np.maximum(0, control_center - control_half) * 100,
        np.minimum(1, control_center + control_half) * 100,
        color="#333333", alpha=0.07, linewidth=0,
    )
    ax_c_radius.plot(
        control["radius_threshold"], control["conditional_retention"] * 100,
        color="#333333", marker="s", markerfacecolor="white", markevery=5, ms=2.0, lw=0.8, label="Positive controls",
    )
    if np.isfinite(p90):
        ax_c_radius.axvline(p90, color=ORANGE, ls="--", lw=0.7, label=f"control P90 {p90:.3f}")
    if np.isfinite(p95):
        ax_c_radius.axvline(p95, color=GRAY, ls=":", lw=0.7, label=f"control P95 {p95:.3f}")
    ax_c_radius.set_xlabel("G3 mismatch threshold", labelpad=1)
    ax_c_radius.set_xlim(0.05, 0.30)
    ax_c_radius.set_ylim(-2, 104)
    ax_c_radius.text(
        0.04, 0.82, "G3 radius mismatch", transform=ax_c_radius.transAxes, fontsize=5.0,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.4},
    )
    ax_c_radius.legend(frameon=False, fontsize=3.9, loc="lower right", labelspacing=0.15, handlelength=1.2)
    style_axis(ax_c_radius)
    fig.text(0.735, 0.94, "C", va="top", ha="left", fontsize=11, fontweight="bold")
    fig.text(0.716, 0.53, "Conditional retention (%)", rotation=90, va="center", ha="center", fontsize=5.8)
    save(fig, "Figure5_revised_screening")


def captions() -> None:
    results = json.loads((RESULTS / "results_for_manuscript.json").read_text(encoding="utf-8"))
    primary = results["primary_candidate_analysis"]
    known = json.loads((RESULTS / "known_control_summary.json").read_text(encoding="utf-8"))
    primary_label = "NSGA-II with prototype floor" if primary["primary_arm"] == "nsga2__floor_on" else "random search with prototype floor"
    text = f"""# Revised figure captions

**Figure 1. Conceptual rationale for model-guided composition-level cathode prioritization.** The retained author-drawn diagram contrasts the slow feedback loop of conventional manual selection, synthesis, and testing with the large-space/slow-testing bottleneck and the present model-guided sequence of surrogate triage, guided proposal, and post-hoc chemistry checks. In the canonical implementation, parsed chemical identity and leakage-resistant evaluation belong to model triage; the matched twenty-seed 2 × 2 benchmark belongs to guided proposal; and G1–G4 plus risk tiers belong to chemistry checks. The endpoint is a working set for independent follow-up, not material validation.

**Figure 2. Canonical composition, measurement context, and template applicability.** (A) Record-multiplicity histogram for 171 parsed canonical compositions; the median is 3 records and the maximum is 116. (B) Temperature, C-rate, and cycle-number distributions; current-rate and cycle axes are logarithmic for display. (C) Measured-capacity distributions for records whose canonical compositions are applicable to one of the 12 declared search templates. Boxes and light family-coloured points summarize records; white diamonds are medians of the canonical-composition medians within each prototype. Compositions outside the declared templates are labeled not applicable rather than forced into a family.

**Figure 3. Deployment-relevant validation hierarchy and distance–error scope check.** (A) R² under record-level cross-validation, exact canonical-composition holdout, the exact 298.15 K/0.5 C/cycle-1 slice of that holdout, and leave-one-structural-family-out validation. (B) Exact leave-one-canonical-composition-out predictions. (C) Each point is one canonical composition (N = 171), represented by its median held-out absolute error and fold-consistent normalized 5NN distance. The weak positive association is statistically inconclusive; distance is not calibrated predictive uncertainty, but weak association is not excluded.

**Figure 4. Matched multi-seed proposal and prototype-floor benchmark.** (A) NSGA-II top-50 median raw and distance-discounted predicted capacity over 25 generations; thin lines are individual seeds and ribbons are mean ± SD. (B) Seed-level post-screen feasible hypervolume for all four factorial arms across twenty matched seeds. (C) Mean ± SD prototype counts in the two floor-on TopK pools; the dashed line marks the imposed minimum of 10. Every proposal used 2,500 objective evaluations per prototype, identical bounds, a shared first batch, common ranking rules, TopK = 500, and the same post-hoc gates.

**Figure 5. Control-anchored screening, prototype-resolved first failure, and threshold sensitivity.** (A) Mean ± SD cumulative retention for the two floor-on twenty-seed proposal arms and retention of {known['template_applicable_controls']} canonically deduplicated, template-applicable positive controls; G3 uses the cross-fitted positive-control P95 reference policy. (B) Prototype-row-normalized first-failure heat maps under P95 for the positive controls and the auditable seed-42 reference TopK ({primary['primary_arm']}); prototype denominators are shown. (C) Complete candidate and control curves for the 0.5–1.5 Li/TM-window-width sweep and the 0.05–0.30 weighted-RMS radius-mismatch sweep. Candidate shading is seed SD; control shading is a Wilson 95% interval. P95 is the reference operating point and P90 defines a nested conservative core. Positive controls estimate retention, not specificity.
"""
    text = text.replace(primary["primary_arm"], primary_label)
    (FIGURES / "figure_captions.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", choices=["1", "2", "3", "4", "5"])
    args = parser.parse_args()
    functions = {"1": figure1, "2": figure2, "3": figure3, "4": figure4, "5": figure5}
    if args.figure:
        setup_style()
        functions[args.figure]()
        return

    # Render each figure in a clean process.  This also isolates Matplotlib's
    # renderer state so a dense rasterized panel cannot affect a later canvas.
    for figure_number in functions:
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--figure", figure_number],
            check=True,
            env={**os.environ, "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl-paper1"))},
        )
    captions()
    print(f"Figures written to {FIGURES}")


if __name__ == "__main__":
    main()
