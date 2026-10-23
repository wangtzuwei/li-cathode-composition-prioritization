"""Family holdout with chemistry-based removal of untemplated compositions (analysis 27).

Each of the 59 untemplated training compositions is judged by stoichiometry and chemistry (CLASSIFICATION
below, with the reason).  A composition judged to belong to a family is removed from the training set when
that family is held out.  The family holdout is then rerun with the original settings
(candidate_checks.family_holdout = extended_validation.leave_one_family_out_validation: CB_PARAMS, thread_count 1,
random_seed 43000 + family index), and the run without removal is repeated as a check.
  main     only the compositions that clearly belong to a family
  variant  main + the six LixMnO2 compositions as layered
New models: models_local/family_holdout_conservative/ (not distributed).  Outputs: outputs/27_*.
"""
from __future__ import annotations

import json

import pandas as pd

import candidate_checks as t
from candidate_checks import OUT, SAVED, log, write_json

MODEL_OUT = t.HERE.parent / "models_local" / "family_holdout_conservative"
LIXMNO2 = ["Li0.19Mn1O2", "Li0.21Mn1O2", "Li0.23Mn1O2", "Li0.28Mn1O2", "Li0.34Mn1O2", "Li0.41Mn1O2"]


def classify(formula: str) -> tuple[str, str]:
    """(family or '', reason) from the canonical formula."""
    if formula == "Li1Mn1.95Si0.05O4":
        return "spinel", "LiM2O4 化學計量（Li : M : O = 1 : 2 : 4），即 LiMn2O4 中 2.5% 的 Mn 由 Si 取代"
    if formula == "Li0.5Mn1P1O4":
        return "olivine", "LiMPO4 化學計量的去鋰態（Li0.5MnPO4），即部分去鋰的 LiMnPO4"
    if formula in LIXMNO2:
        return "", ("MO2 化學計量、Li 0.19–0.41；組成本身無法區分層狀（去鋰的 LiMnO2 或 birnessite 型）與隧道型 MnO2 "
                    "（α／γ-MnO2 嵌鋰）等結構，不列為明確屬於某家族（另做層狀變體）")
    if "B1O3" in formula:
        return "", "LiMBO3 硼酸鹽，不屬於三個家族"
    if formula == "Li2S1":
        return "", "硫化物，不屬於三個家族"
    if "V" in formula and "P3O12" in formula or "P2.9" in formula or "P2.85" in formula or "P2.99" in formula:
        return "", "Li3V2(PO4)3 型（NASICON 磷酸鹽），不屬於三個家族"
    if "Si" in formula:
        if formula == "Li1Mn0.8Si1Ni0.2O4":
            return "", "LiMSiO4（Li : M : Si : O = 1 : 1 : 1 : 4）；雖與 LiMPO4 同為 1 : 1 : 1 : 4，但以 Si 取代 P 的 olivine 不是已知的明確歸屬，不列入"
        return "", "Li2MSiO4 型正矽酸鹽（含部分 P 或 Al 取代），四面體結構，不屬於三個家族"
    return "", "未分類"


def main() -> None:
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    t.MODEL_OUT = MODEL_OUT
    features, y, raw = t.training_table()
    assign = pd.read_csv(SAVED / "results" / "known_control_prototype_assignment.csv")
    templated = assign.dropna(subset=["proto_name"]).set_index("canonical_key")["proto_name"].astype(str)
    family_by_key = templated.map(t.STRUCTURAL_FAMILY)
    untemplated = assign[assign["proto_name"].isna()].sort_values("canonical_formula")
    rows = []
    for r in untemplated.itertuples(index=False):
        family, reason = classify(r.canonical_formula)
        rows.append({"canonical_formula": r.canonical_formula, "formula_as_recorded": r.Formula,
                     "canonical_key": r.canonical_key, "n_records": int(r.n_records),
                     "release_rule": r.prototype_assignment_rule, "family": family, "reason": reason})
    table = pd.DataFrame(rows)
    if (table["reason"] == "未分類").any() or len(table) != 59:
        raise RuntimeError("classification incomplete")
    table.to_csv(OUT / "27_untemplated_classification.csv", index=False, encoding="utf-8-sig")

    main_drop = {f: table.loc[table["family"].eq(f), "canonical_key"].tolist() for f in ["layered", "olivine", "spinel"]}
    variant_drop = dict(main_drop)
    variant_drop["layered"] = main_drop["layered"] + table.loc[table["canonical_formula"].isin(LIXMNO2), "canonical_key"].tolist()
    runs = {}
    for tag, drop in [("27_original", {}), ("27_main", main_drop), ("27_variant_LixMnO2_layered", variant_drop)]:
        _, summary = t.family_holdout(features, y, raw, family_by_key, drop, tag)
        (OUT / f"06F_{tag}_predictions.csv").replace(OUT / f"{tag}_predictions.csv")
        runs[tag] = {**summary, "removed": {f: table.set_index("canonical_key").loc[k, "canonical_formula"].tolist()
                                            for f, k in drop.items()}}
        log(f"{tag}: overall R2 {summary['overall']['r2']:.4f}")
    saved = json.loads((SAVED / "results" / "leave_one_family_out_validation.json").read_text(encoding="utf-8"))
    write_json("27_family_holdout_summary.json", {"saved_overall": saved["overall"], "runs": runs})


if __name__ == "__main__":
    main()
