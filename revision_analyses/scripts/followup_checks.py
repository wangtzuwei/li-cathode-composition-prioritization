"""Follow-up analyses 1-4 for analysis 06b.

  1  external comparison: G3 (weighted RMS), cation/anion ratio and strict
     budget for synthesized Cr/F Li-rich materials (literature/) and
     the compositions of manuscript references [63]-[65]
  2  cation/anion ratio of the 112 positive controls by template, vs C's limits
  3  per-composition detail of B's leave-one-out retrains (reads 06B outputs)
  4  F dose response: drop the 5/10/20 untemplated compositions nearest to the
     held-out family and rerun the family holdout (retrain)

New models go to models_local/followup_checks/, tables to outputs/06b*_*.
No saved model or release file is overwritten, and no search is run.

Usage:  python followup_checks.py [1 2 3 4]   (default: all)
"""
from __future__ import annotations

import json
import re
import sys
import time

import numpy as np
import pandas as pd
from pymatgen.core import Composition

import candidate_checks as t
from candidate_checks import OUT, SAVED, SEVEN, SITE_LIMIT, TOL, class_of, ssb, strict_budget

LITERATURE = t.HERE.parent / "literature" / "Cr_F_LiRich_primary_verified_samples.csv"
MODEL_OUT = t.HERE.parent / "models_local" / "followup_checks"
LAYERED_P95 = 0.0659489882206248
POOLED_P95, _ = t.p95_p90()
_SEVEN_VALUES = [ssb.radius_mismatch_metrics(f, "Li2MnO3")[0] for f in SEVEN]
SEVEN_G3 = (min(_SEVEN_VALUES), max(_SEVEN_VALUES))

# Manuscript references [63]-[65] (formulas as listed in the paper).
REFERENCE_ROWS = [
    ("[63] Pan et al. 2002", "Li[Li0.2Cr0.4Mn0.4]O2", "Li1.2Cr0.4Mn0.4O2"),
    ("[64] Dong et al. 2015", "Li2MnO3-xFx, x = 0", "Li2MnO3"),
    ("[64] Dong et al. 2015", "Li2MnO3-xFx, x = 0.015", "Li2MnO2.985F0.015"),
    ("[64] Dong et al. 2015", "Li2MnO3-xFx, x = 0.03", "Li2MnO2.97F0.03"),
    ("[64] Dong et al. 2015", "Li2MnO3-xFx, x = 0.045", "Li2MnO2.955F0.045"),
    ("[65] Huang et al. 2021", "Li1.2Ni0.13Co0.13Mn0.54O2 (undoped base in the title)", "Li1.2Ni0.13Co0.13Mn0.54O2"),
]


def literature_formula(nominal: str, x: str) -> str:
    """Turn the paper's nominal formula with an x parameter into a plain formula."""
    xv = float(x)
    base = re.sub(r"\s*\(x\s*=.*\)$", "", nominal).replace("−", "-").strip()
    if "O2(1-x)F2x" in base:
        return base.replace("O2(1-x)F2x", f"O{2 * (1 - xv):g}F{2 * xv:g}") if xv else base.replace("O2(1-x)F2x", "O2")
    match = re.search(r"O([\d.]+)-xFx$", base)
    if match:
        oxygen = float(match.group(1))
        tail = f"O{oxygen - xv:g}F{xv:g}" if xv else f"O{oxygen:g}"
        return base[:match.start()] + tail
    return base


def g3_row(label: str, nominal: str, formula: str, extra: dict[str, object]) -> dict[str, object]:
    proto, rule = ssb.assign_known_prototype(formula)
    rms_assigned = ssb.radius_mismatch_metrics(formula, proto)[0] if proto else np.nan
    rms_li2mno3 = ssb.radius_mismatch_metrics(formula, "Li2MnO3")[0]
    strict = strict_budget(formula)
    value = rms_assigned if proto else rms_li2mno3
    return {
        "source": label, "nominal": nominal, "Formula": formula, **extra,
        "assigned_template": proto, "assignment_rule": rule,
        "G3_assigned_template": rms_assigned, "G3_Li2MnO3_template": rms_li2mno3,
        "le_layered_P95": bool(value <= LAYERED_P95), "le_pooled_P95": bool(value <= POOLED_P95),
        "relative_to_seven": ("below" if value < SEVEN_G3[0] else "above" if value > SEVEN_G3[1] else "within"),
        "cation_to_anion": strict["cation_to_anion_ratio"],
        "above_layered_site_limit": bool(strict["cation_to_anion_ratio"] > 1.0 + TOL),
        "strict_budget_mAh_g": strict["strict_budget_mAh_g"],
        "original_G4_budget_mAh_g": ssb.redox_budget(formula)[1],
    }


def item_1() -> None:
    t.log("1: external comparison")
    lit = pd.read_csv(LITERATURE, encoding="utf-8-sig")
    rows = []
    for _, r in lit.iterrows():
        nominal = r["論文中名義化學式_保留原文參數寫法"]
        formula = literature_formula(nominal, r["摻雜量_原文x"])
        rows.append(g3_row(f"{r['作者_簡稱']} {r['年份']}", nominal, formula, {
            "DOI": r["DOI"], "sample": r["樣品名"], "first_charge": r["首圈充電容量_mAh_g"],
            "first_discharge": r["首圈放電容量_mAh_g"], "later_discharge": r["活化後或第N圈放電容量_mAh_g"],
            "later_cycle": r["活化後或第N圈的循環次數"]}))
    for label, nominal, formula in REFERENCE_ROWS:
        rows.append(g3_row(label, nominal, formula, {}))
    for formula in SEVEN:
        rows.append(g3_row("7 candidates", formula, formula, {}))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "06b1_external_g3.csv", index=False)
    t.write_json("06b1_summary.json", {"layered_P95": LAYERED_P95, "pooled_P95": POOLED_P95,
                                       "seven_G3_range": SEVEN_G3,
                                       "skipped": "[65] Cr/F co-doped formula not found (abstract unavailable via "
                                                  "Springer, Semantic Scholar and Crossref on 2026-10-23)",
                                       "rows": table.to_dict(orient="records")})


def item_2() -> None:
    t.log("2: controls cation/anion by template")
    controls = pd.read_csv(SAVED / "results" / "known_control_screened.csv")
    controls["class"] = controls["proto_name"].map(class_of)
    controls["cation_to_anion"] = controls["Formula"].map(t.cation_anion)
    controls["limit"] = controls["class"].map(SITE_LIMIT)
    controls["above_limit"] = controls["cation_to_anion"] > controls["limit"] + TOL
    controls[["proto_name", "class", "Formula", "cation_to_anion", "limit", "above_limit"]].to_csv(
        OUT / "06b2_controls_ratio.csv", index=False)
    summary = controls.groupby("proto_name", sort=True).agg(
        cls=("class", "first"), limit=("limit", "first"), n=("Formula", "size"), n_above=("above_limit", "sum"),
        ratio_min=("cation_to_anion", "min"), ratio_p25=("cation_to_anion", lambda s: s.quantile(0.25)),
        ratio_median=("cation_to_anion", "median"), ratio_p75=("cation_to_anion", lambda s: s.quantile(0.75)),
        ratio_max=("cation_to_anion", "max")).reset_index()
    summary["above_formulas"] = summary["proto_name"].map(
        controls[controls["above_limit"]].groupby("proto_name").apply(
            lambda p: "; ".join(f"{f} ({v:.4f})" for f, v in zip(p["Formula"], p["cation_to_anion"]))))
    summary.to_csv(OUT / "06b2_controls_ratio_by_template.csv", index=False)


def item_3() -> None:
    t.log("3: B leave-one-out detail")
    effects = pd.read_csv(OUT / "06B_effects_by_scenario.csv")
    single = effects[effects["scenario"].str.startswith("drop_")
                     & ~effects["scenario"].isin(["drop_all_Cr", "drop_all_F"])].copy()
    single["dropped"] = single["scenario"].str.replace("drop_", "", regex=False)
    influence = single.groupby("dropped").agg(
        mean_abs_candidate_change=("candidate_change_vs_baseline", lambda s: s.abs().mean()),
        max_abs_candidate_change=("candidate_change_vs_baseline", lambda s: s.abs().max()),
        mean_candidate_change=("candidate_change_vs_baseline", "mean"),
        mean_abs_Cr_effect_change=("Cr_effect_change", lambda s: s.abs().mean()),
        mean_abs_F_effect_change=("F_effect_change", lambda s: s.abs().mean()),
        n_not_above_all_three=("above_all_three", lambda s: int((~s).sum())),
    ).reset_index()
    for col in ["mean_abs_candidate_change", "mean_abs_Cr_effect_change", "mean_abs_F_effect_change"]:
        influence[f"rank_{col}"] = influence[col].rank(ascending=False, method="min").astype(int)
    influence.sort_values("mean_abs_candidate_change", ascending=False).to_csv(
        OUT / "06b3_influence_by_composition.csv", index=False)
    single[["dropped", "candidate", "candidate_pred", "candidate_change_vs_baseline", "Cr_effect_a_minus_c",
            "F_effect_b_minus_c", "above_a", "above_b", "above_c"]].to_csv(OUT / "06b3_detail.csv", index=False)


def item_4() -> None:
    t.log("4: F dose response")
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    features, y, raw = t.training_table()
    assign = pd.read_csv(SAVED / "results" / "known_control_prototype_assignment.csv")
    templated = assign.dropna(subset=["proto_name"]).set_index("canonical_key")["proto_name"].astype(str)
    family_by_key = templated.map(t.STRUCTURAL_FAMILY)
    untemplated = assign[assign["proto_name"].isna()][["canonical_key", "canonical_formula", "n_records"]]
    ranking = []
    for family in ["layered", "olivine", "spinel"]:
        members = list(family_by_key[family_by_key.eq(family)].index)
        M = np.array([t.frac_vector(k) for k in members])
        for u in untemplated.itertuples(index=False):
            d = np.abs(M - t.frac_vector(u.canonical_key)).sum(axis=1)
            ranking.append({"family": family, "canonical_key": u.canonical_key, "formula": u.canonical_formula,
                            "n_records": int(u.n_records), "min_distance": float(d.min()),
                            "nearest_member": assign.set_index("canonical_key").loc[members[int(np.argmin(d))],
                                                                                    "canonical_formula"]})
    ranking = pd.DataFrame(ranking).sort_values(["family", "min_distance", "formula"]).reset_index(drop=True)
    ranking["rank"] = ranking.groupby("family").cumcount() + 1
    ranking.to_csv(OUT / "06b4_untemplated_ranking.csv", index=False)

    results = {}
    original_mask_check = None
    for n_drop in [0, 5, 10, 20]:
        drop = {f: ranking[(ranking["family"].eq(f)) & ranking["rank"].le(n_drop)]["canonical_key"].tolist()
                for f in ["layered", "olivine", "spinel"]}
        tag = f"nearest{n_drop:02d}"
        t.MODEL_OUT = MODEL_OUT  # family_holdout saves into candidate_checks.MODEL_OUT
        combined, summary = t.family_holdout(features, y, raw, family_by_key, drop, tag)
        (OUT / f"06F_{tag}_predictions.csv").replace(OUT / f"06b4_{tag}_predictions.csv")
        if n_drop == 0:
            original_mask_check = summary["overall"]["r2"]
        results[tag] = {"n_drop_per_family": n_drop, **summary,
                        "dropped": {f: ranking[(ranking["family"].eq(f)) & ranking["rank"].le(n_drop)][
                            ["formula", "n_records", "min_distance"]].to_dict(orient="records")
                                    for f in ["layered", "olivine", "spinel"]}}
    saved = json.loads((SAVED / "results" / "leave_one_family_out_validation.json").read_text(encoding="utf-8"))
    t.write_json("06b4_summary.json", {"saved_overall_r2": saved["overall"]["r2"],
                                       "replicate_r2": original_mask_check, "runs": results})


ITEMS = {"1": item_1, "2": item_2, "3": item_3, "4": item_4}

if __name__ == "__main__":
    for item in sys.argv[1:] or ["1", "2", "3", "4"]:
        start = time.time()
        ITEMS[item]()
        t.log(f"{item} finished in {time.time() - start:.0f} s")
