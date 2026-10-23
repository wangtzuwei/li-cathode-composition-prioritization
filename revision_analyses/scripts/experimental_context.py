"""24: predictions next to measured values for synthesized Cr- or F-substituted Li2MnO3 (analysis 24; see revision_analyses/README.md).

Scope: compositions obtained nominally from Li2MnO3 by Cr-for-Mn or F-for-O substitution (Li : (Mn + Cr) = 2 : 1,
or the Li[Li(1-x)/3 Mn(2-2x)/3 Crx]O2 solid-solution line), plus the undoped Li2MnO3 reference of the same paper.
  training data   Dong 2015 (10.1039/C4TA02924K): all 15 training records of the Li2MnO3 template
                  (literature/records_to_check.csv).  Predictions: saved full model at each record's own
                  feature row (in-sample) and the saved composition-holdout OOF prediction (out-of-sample).
  external        Kim 2013 from the v2 sample table (outputs/08b_predictions.csv, 1C = 200 / 250 / 300 mAh/g).
                  Li1.2Cr0.4Mn0.4O2 (Ammundsen 2002; Pan 2002) and Li2MnO2F (Blumenhofer 2023) are listed with the
                  capacity column empty: no capacity is recorded in the local literature tables.
  candidates      the 7 candidates and the Li = 2 member, full model at 298.15 K, cycle 1 and the C-rates used above.
Saved model only; nothing is trained.  Outputs: outputs/24_*.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from pymatgen.core import Composition

from candidate_checks import HERE, OUT, SAVED, SEVEN, log, ssb, write_json

LIT = HERE.parent / "literature"
LI2 = "Li2Mn0.7Cr0.3O2.865F0.135"
CANDIDATE_RATES = [0.04, 0.05, 0.1, 0.5, 1.0]
NO_CAPACITY = [("Ammundsen 2002；Pan 2002（[63]）", "Li1.2Cr0.4Mn0.4O2", "10.1149/1.1456535"),
               ("Blumenhofer 2023（DRX）", "Li2MnO2F", "10.1021/acsmaterialsau.2c00064")]


def predict(ev: ssb.CandidateEvaluator, formula: str, rate: float, cycle: float, temperature: float = 298.15) -> float:
    comp = Composition(formula)
    row = ev._composition_row({"comp": comp.get_el_amt_dict(), "Formula": formula, "canonical_key": ssb.canonical_key(comp)})
    X = pd.DataFrame([row]).reindex(columns=ev.comp_columns)
    X["temperature_K"] = temperature
    X["log1p_current_rate_C"] = math.log1p(rate)
    X["log1p_cycle_number"] = math.log1p(cycle)
    X["log1p_current_rate_x_cycle"] = math.log1p(rate * cycle)
    return float(ev.model.predict(X.reindex(columns=ev.feature_columns), thread_count=1)[0])


def main() -> None:
    ev = ssb.CandidateEvaluator()
    feats = pd.read_csv(SAVED / "data" / "features_composition_context.csv")
    oof = pd.read_csv(SAVED / "results" / "oof_canonical_composition_heldout_all_models.csv")
    oof = oof[oof["model"].eq("catboost__composition_plus_context")].set_index("row_id")
    rec = pd.read_csv(LIT / "records_to_check.csv", encoding="utf-8-sig")
    dong = rec[rec["proto_name"].eq("Li2MnO3")].sort_values(["canonical_formula", "cycle_value", "current_rate"])
    rows = []
    for r in dong.itertuples(index=False):
        x = feats.iloc[[int(r.row_id)]]
        in_sample = float(ev.model.predict(x, thread_count=1)[0])
        if abs(predict(ev, r.canonical_formula, r.current_rate, r.cycle_value, r.temperature) - in_sample) > 1e-6:
            raise RuntimeError(f"feature row {r.row_id} not reproduced from the formula")
        rows.append({"source": "訓練資料", "paper": "Dong 2015", "DOI": r.DOI, "sample": r.canonical_formula,
                     "model_formula": r.canonical_formula, "C_rate": r.current_rate, "cycle": r.cycle_value,
                     "temperature_K": r.temperature, "condition_note": "訓練資料中記錄的條件",
                     "specifier": r.BDE_Specifier if isinstance(r.BDE_Specifier, str) else "",
                     "measured": r.Value, "pred_full_in_sample": in_sample,
                     "pred_oof_composition_holdout": float(oof.loc[int(r.row_id), "y_pred"]),
                     "y_true_check": float(oof.loc[int(r.row_id), "y_true"])})
    kim = pd.read_csv(OUT / "08b_predictions.csv", encoding="utf-8-sig")
    kim = kim[kim["paper"].eq("Kim") & kim["scenario"].str.startswith("首圈放電") & kim["sample"].str.contains("Li2MnO3")
              & ~kim["sample"].str.contains("Al|Fe|Zn")]
    for (sample, formula), part in kim.groupby(["sample", "Formula"], sort=False):
        p = part.assign(C_rate=part["C_rate"].round(6)).set_index("C_rate")["pred_full"]
        rows.append({"source": "外部文獻", "paper": "Kim 2013", "DOI": "10.1007/s10832-012-9778-4", "sample": sample,
                     "model_formula": formula, "C_rate": 10 / 250, "cycle": 1, "temperature_K": 298.15,
                     "condition_note": "10 mA/g，1C 未定義：0.05／0.04／0.0333 C（1C = 200／250／300 mAh/g）；溫度寫 room temperature",
                     "specifier": "首圈放電", "measured": part["measured"].iloc[0],
                     "pred_full_1C200": float(p.loc[round(10 / 200, 6)]), "pred_full_1C250": float(p.loc[round(10 / 250, 6)]),
                     "pred_full_1C300": float(p.loc[round(10 / 300, 6)])})
    for paper, formula, doi in NO_CAPACITY:
        rows.append({"source": "外部文獻（本機文獻表沒有容量數值）", "paper": paper, "DOI": doi, "sample": formula,
                     "model_formula": formula, "C_rate": 0.5, "cycle": 1, "temperature_K": 298.15,
                     "condition_note": "搜尋時的固定條件", "measured": np.nan,
                     "pred_full_search_context": predict(ev, formula, 0.5, 1)})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "24_experimental_rows.csv", index=False, encoding="utf-8-sig")

    cand = []
    for f in SEVEN + [LI2]:
        cand.append({"source": "模型候選" if f in SEVEN else "Li = 2 成員", "Formula": f,
                     **{f"pred_{rate:g}C_cycle1": predict(ev, f, rate, 1) for rate in CANDIDATE_RATES},
                     "pred_0.1C_cycle100": predict(ev, f, 0.1, 100), "pred_1C_cycle100": predict(ev, f, 1.0, 100)})
    cand = pd.DataFrame(cand)
    cand.to_csv(OUT / "24_candidates.csv", index=False, encoding="utf-8-sig")
    write_json("24_summary.json", {"dong_records": int(len(dong)), "prediction_clip": ev.prediction_clip,
                                   "oof_y_true_matches": bool(np.allclose(table.loc[table["paper"].eq("Dong 2015"), "measured"],
                                                                          table.loc[table["paper"].eq("Dong 2015"), "y_true_check"]))})
    log("done")
    pd.set_option("display.width", 300)
    print(table.drop(columns=["DOI", "condition_note"]).round(1).to_string())
    print(cand.round(1).to_string())


if __name__ == "__main__":
    main()
