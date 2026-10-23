"""Test-condition extraction (analysis 14; see revision_analyses/README.md).

Step 1 (extract): for every paper that 09 links to a single source (tiers A, B, C, G) among the records of
literature/records_to_check.csv, read the local PDF (pdftotext -layout, page by page) and collect
every text snippet matching: a voltage window, an electrolyte, a current density / C-rate and the 1C
definition, a temperature, an activation step, and each record's capacity value.  Candidates are written to
outputs/14_candidates.json with page numbers.  Nothing is inferred.
Step 2 (curate): the values confirmed by reading those snippets are listed in CURATED below, each with its
page and quoted fragment; fields without a confirmed snippet are "未找到".
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from candidate_checks import OUT, log, write_json

DATA_ROOT = Path(os.environ.get("PAPER_PDF_ROOT", "PAPER_PDF_ROOT_not_set"))  # article PDFs, not distributed
LIT = Path(__file__).resolve().parents[1] / "literature"
SINGLE = ["A", "B", "C", "G"]

PATTERNS = {
    "voltage": re.compile(r"(\d\.\d{1,2})\s*(?:V)?\s*(?:–|-|−|~|to|and)\s*(\d\.\d{1,2})\s*V", re.I),
    "electrolyte": re.compile(r"LiPF\s?6|LiClO\s?4|LiBF\s?4|LiTFSI|ethylene carbonate|\bEC\b", re.I),
    "c_definition": re.compile(r"1\s*C\s*(?:=|is|corresponds|was defined|equal)[^.]{0,40}?\d+(?:\.\d+)?\s*mA", re.I),
    "current": re.compile(r"\d+(?:\.\d+)?\s*mA\s*(?:g|cm)\s*[-−–]?\s*1|C\s*/\s*\d+|\b\d+(?:\.\d+)?\s*C\b(?!\s*(?:°|o))", re.I),
    "temperature": re.compile(r"room temperature|ambient temperature|\d{2}\s*°\s*C|\d{2}\s*oC|\b\d{3}\s*K\b", re.I),
    "activation": re.compile(r"activat|formation cycle|pre-?cycl", re.I),
}


def pages(pdf: Path) -> list[str]:
    text = subprocess.run(["pdftotext", "-enc", "UTF-8", "-layout", str(pdf), "-"], capture_output=True,
                          text=True, encoding="utf-8", errors="replace").stdout
    return text.split("\f")


def snippet(line: str, m: re.Match, width: int = 90) -> str:
    s = max(0, m.start() - width)
    return " ".join(line[s:m.end() + width].split())


def extract() -> dict:
    rec = pd.read_csv(LIT / "records_to_check.csv", encoding="utf-8-sig")
    rec["tier"] = rec["trace_status"].str[:2].str.strip()
    pdfs = pd.read_csv(OUT / "09_pdf_dois.csv", encoding="utf-8-sig")
    pdfs["d"] = pdfs["DOI"].astype(str).str.lower()
    out = {}
    for doi, part in rec[rec["tier"].isin(SINGLE)].groupby("DOI"):
        files = pdfs[pdfs["d"].eq(doi.lower())]["file"].tolist()
        # prefer the numbered verification copy, then any other copy
        files = sorted(files, key=lambda f: ("20260102" in f, f))
        pdf = DATA_ROOT / files[0].replace("\\", "/")
        pg = pages(pdf)
        found = {k: [] for k in PATTERNS}
        found["capacity_values"] = []
        values = sorted({f"{v:g}" for v in part["Value"]})
        refs_started = False
        for p, page in enumerate(pg, 1):
            for line in page.splitlines():
                if re.match(r"^\s*(References|REFERENCES|Notes and references)\s*$", line):
                    refs_started = True
                if refs_started:
                    continue
                for k, pat in PATTERNS.items():
                    for m in pat.finditer(line):
                        found[k].append({"page": p, "text": snippet(line, m)})
                for v in values:
                    for m in re.finditer(rf"(?<![\d.]){re.escape(v)}(?![\d])", line):
                        found["capacity_values"].append({"page": p, "value": v, "text": snippet(line, m, 70)})
        out[doi] = {"pdf": str(pdf.relative_to(DATA_ROOT)), "n_pages": len(pg), "n_records": int(len(part)),
                    "values": values, "found": found}
        log(f"{doi}: {len(pg)} pages")
    (OUT / "14_candidates.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


# ----------------------------------------------------------------------------- curated (read from the snippets)
NF = "未找到"
NO_ACT = "未找到明確的活化步驟"
CURATED = {
    "10.1016/j.apsusc.2013.09.044": {
        "voltage": ("3.0", "4.35", "p4: “tester at room temperature over the voltage range 3.0–4.35 V”"),
        "electrolyte": "p4: “1 M of LiPF6 … dissolved in EC:DMC = 1:1 (v/v) was used as the electrolyte”",
        "current": "p4: “measured at 0.2 C and 1 C at room temperature”", "c_def": NF,
        "temperature": "p4: “at room temperature”", "activation": NO_ACT},
    "10.1016/j.electacta.2009.05.041": {
        "voltage": ("3.0", "4.8", "p2: “at 0.2C rate in the range of 3.0–4.8 V”"),
        "electrolyte": "p2: “1 M LiPF6 in EC:DMC (1:1 vol.%)”",
        "current": "p2: “0.2C rate”; p4: “at various charge–discharge rates”",
        "c_def": "p2: “1C rate corresponds to a current rate of 180 mA g−1”",
        "temperature": "p4: “at room temperature”", "activation": NO_ACT},
    "10.1016/j.electacta.2013.01.159": {
        "voltage": ("2.5", "4.7", "p2: “galvanostatically between 2.5 and 4.7 V at 25 °C”；倍率測試的圖說（Fig. 4）寫 “at different rates in the voltage range of 2.5–4.6 V”"),
        "voltage_rate_test": ("2.5", "4.6 或 4.7", "Methods 寫 2.5–4.7 V，Fig. 4 圖說寫 2.5–4.6 V"),
        "electrolyte": "p2: “1 M LiPF6 dissolved in EC/DMC/DEC (1:1:1 in volume rate)”",
        "current": "p2: “at different rates”；p4: “at 0.2 C”、“0.5 C … 1 C … 2 C … 3 C … 5 C”", "c_def": NF,
        "temperature": "p2: “at 25 °C”", "activation": NO_ACT + "（p3 只提到 Li2MnO3 成分的活化現象）"},
    "10.1016/j.jpowsour.2014.02.011": {
        "voltage": ("2.0", "4.8", "p3: “between 2.0 and 4.8 V”"),
        "electrolyte": "p3: “1 M LiPF6 in ethylene carbonate (EC)–dimethyl carbonate (DMC) (1:1 in volume)”",
        "current": "p3: “densities from 20 mA g−1 to 2000 mA g−1”", "c_def": NF,
        "temperature": "p3: “at room temperature”", "activation": NO_ACT + "（p4、p6 只提到 Li2MnO3 區域的活化現象）"},
    "10.1016/j.jpowsour.2016.09.028": {
        "voltage": ("2.0", "4.8", "p3: “various current rates (1C = 200 mA g−1) between 2.0 V and 4.8 V”"),
        "electrolyte": "p3: “1 M LiPF6 dissolved in ethylene carbonate, dimethyl carbonate and diethyl carbonate (1:1:1, vol%)”",
        "current": "p7: “from 0.1 C to 5 C”", "c_def": "p3: “1C = 200 mA g−1”",
        "temperature": NF, "activation": NO_ACT + "（p5、p6 只提到 Li2MnO3 的活化現象）"},
    "10.1016/j.jssc.2017.08.039": {
        "voltage": ("3.0", "5.0", "p2: “after 100 cycles between 3.0 and 5.0 V”"),
        "electrolyte": "p2: “1.0 mol L−1 LiPF6 in a mixture of dimethyl carbonate (DMC)/ethylene carbonate (EC) (v/v, 1/1)” 並加入 tris(trimethylsilyl) 添加劑",
        "current": "p7: “at 0.1 C, 0.2 C, 0.5 C and 1 C rates”", "c_def": "p5: “0.1 C (1 C = 170 mA h g−1)”",
        "temperature": NF + "（p8 的 298 K 是阻抗分析用的溫度）", "activation": NO_ACT},
    "10.1016/j.nanoen.2016.01.014": {
        "voltage": ("2.0", "4.8", "p2: “in a voltage range of 2.0–4.8 V at room temperature and 60 °C”"),
        "electrolyte": "p2: “1 M LiPF6 in EC:DMC (1:1 in volume)”",
        "current": "p5: “0.1 C (20 mA g−1)”；p6: “1.5 C”", "c_def": "p6: “0.1 C (20 mA g−1)”（即 1C = 200 mA g−1）",
        "temperature": "p2: “at room temperature and 60 °C”；p5: “when the test temperature increases from 25 to 60 °C”",
        "activation": NO_ACT + "（p5 提到 60 °C 測試前先做 30 圈：“30 charge–discharge cycles were conducted before high-temperature cycle life”）"},
    "10.1039/C4RA11894D": {
        "voltage": ("2.0", "4.8", "p3: “with current density of 20 mA g−1 (0.1 C) between 2.0 and 4.8 V”"),
        "electrolyte": "p2: “1 M LiPF6 dissolved in a mixture of ethylene carbonate (EC) and dimethyl carbonate (DMC)”",
        "current": "p3: “20 mA g−1 (0.1 C)”", "c_def": "p1: “1 C corresponds to 200 mA g−1”",
        "temperature": "p3: “at room temperature”", "activation": NO_ACT},
    "10.1039/C4TA01217H": {
        "voltage": ("2.0", "4.8", "p3: “in the voltage range of 2.0–4.8 V”"),
        "electrolyte": "p3: “1 M LiPF6 electrolyte in ethylene carbonate–diethyl carbonate (EC–DEC, 1:1 by volume)”",
        "current": "p1: “at a higher rate of 2 C”", "c_def": NF,
        "temperature": NF, "activation": NO_ACT + "（p8 只在討論中提到 pre-cycling）"},
    "10.1039/C4TA02924K": {
        "voltage": ("2.0", "4.6", "p2: “at room temperature between 2.0 and 4.6 V vs. Li/Li+”"),
        "electrolyte": "p2: “1 M LiPF6 dissolved in ethylene carbonate–diethyl carbonate–ethyl methyl carbonate with a 1:1:1 volume ratio”",
        "current": "p1: “at 0.05 C”、“at 1 C”；p8: “at 16 C”", "c_def": NF,
        "temperature": "p2: “at room temperature”", "activation": NO_ACT + "（文中多處討論 Li2MnO3 的活化平台）"},
    "10.1039/C7RA03028B": {
        "voltage": ("1.5", "4.5", "p2: “voltage range between 1.5 and 4.5 V”"),
        "electrolyte": "p2: “1 M LiPF6 in EC:DMC (1:1 vol%)”",
        "current": "p1: “at 0.1C”；p4: “2C and 5C”", "c_def": "p3: “at 0.1C (1C = 220 mA g−1)”",
        "temperature": "p2: “performed at room temperature (25 °C)”", "activation": NO_ACT},
}

# Cycle number read by hand from the wider text around each value (the automatic ±70-character check
# mis-assigned several of these; see analysis 14 section 1.3).
MANUAL_CYCLE = {
    ("10.1016/j.nanoen.2016.01.014", "280.8"): ("首圈（原文）", "p5: “The discharge capacity of the 1st cycle is 280.8 mA h g−1”"),
    ("10.1016/j.nanoen.2016.01.014", "172.3"): ("第 200 圈（原文）", "p5: “its discharge capacity after 200 cycles is still larger than 172.3 mA h g−1”"),
    ("10.1016/j.nanoen.2016.01.014", "189.3"): ("未找到（原文為 3.5 C 下的「最高放電容量」，未寫圈數）", "p5: “the highest discharge capacity of the nanoplate cathode is as high as 189.3 mA h g−1”"),
    ("10.1016/j.nanoen.2016.01.014", "303.4"): ("未找到（原文為 60 °C 下的「最高放電容量」，未寫圈數）", "p6: “At 60 °C, the highest discharge capacities … at 0.5, 1.5 and 3.5 C are 303.4, 259.3 and 213.7 mA h g−1”"),
    ("10.1016/j.nanoen.2016.01.014", "259.3"): ("未找到（原文為 60 °C 下的「最高放電容量」，未寫圈數）", "p6: “At 60 °C, the highest discharge capacities … at 0.5, 1.5 and 3.5 C are 303.4, 259.3 and 213.7 mA h g−1”"),
    ("10.1016/j.nanoen.2016.01.014", "213.7"): ("未找到（原文為 60 °C 下的「最高放電容量」，未寫圈數）", "p6: “At 60 °C, the highest discharge capacities … at 0.5, 1.5 and 3.5 C are 303.4, 259.3 and 213.7 mA h g−1”"),
    ("10.1039/C4RA11894D", "180.7"): ("第 50 圈（原文）", "p1: “maintains 180.7 mA h g−1 at 1 C rate after 50 cycles”"),
    ("10.1039/C7RA03028B", "196.3"): ("首圈（原文）", "p3: “exhibits an initial discharge specific capacity of 196.3 mA h g−1 at 0.1C”"),
    ("10.1039/C7RA03028B", "161.9"): ("未找到（原文只寫 1C 下的放電比容量，未寫圈數）", "p4: “exhibits higher discharge specific capacity of 161.9 mA h g−1 with higher capacity retention of 95% after 50 cycles at 1C”"),
    ("10.1016/j.jssc.2017.08.039", "144"): ("首圈（原文）", "p1: “LiCo0.94Cr0.04PO4/C delivers an initial discharge capacity of 144 mA h g−1 at 0.1 C”"),
    ("10.1039/C4TA02924K", "299"): ("首圈（原文）", "p1: “a first discharge capacity of 299 mA h g−1 with a high energy density of 934 W h kg−1 at 0.05 C”"),
    ("10.1039/C4TA02924K", "213"): ("首圈（依上下文：接在初次充電的描述之後）", "p5: “more efficient activation in the initial charging. Thereafter, very high discharge capacities of above 285 mA h g−1 at 0.05 C are delivered by F-substituted samples in contrast with 213 mA h g−1 for Li2MnO3”"),
    ("10.1039/C4TA02924K", "183"): ("未找到（原文為 1 C 下的「最高放電容量」，未寫圈數）", "p8: “The 1.0% sample delivers the highest discharge capacity of 183 mA h g−1 with a retention of 92% after 100 cycles”"),
    ("10.1039/C4TA02924K", "94.7"): ("未找到（原文未寫圈數）", "p6: “pristine Li2MnO3 delivers a charge–discharge capacity of only 80/94.7 mA h g−1 at 1 C”"),
    ("10.1016/j.electacta.2009.05.041", "154.9"): ("未找到（原文未寫圈數）", "p3: “the Li3V1.5Cr0.5(PO4)3/C sample only give a discharge capacity of 154.9 mAh g−1 (86.7% of initial capacity of the undoped sample) at 0.2C”"),
}


def cycle_from_snippets(snips: list[dict]) -> tuple[str, str]:
    if not snips:
        return "數值未在文字中出現（可能取自圖表）", ""
    for s in snips:
        t = s["text"]
        m = re.search(r"after\s+(\d+)\s+cycles|(\d+)(?:st|nd|rd|th)\s+cycle", t, re.I)
        if m:
            return f"第 {m.group(1) or m.group(2)} 圈（原文）", f"p{s['page']}: “{t}”"
        if re.search(r"initial|first|1st", t, re.I):
            return "首圈（原文）", f"p{s['page']}: “{t}”"
    s = snips[0]
    return "未找到（文字中有此數值，但沒寫圈數）", f"p{s['page']}: “{s['text']}”"


def build(cands: dict) -> pd.DataFrame:
    rec = pd.read_csv(LIT / "records_to_check.csv", encoding="utf-8-sig")
    rec["tier"] = rec["trace_status"].str[:2].str.strip()
    rows = []
    for r in rec.itertuples(index=False):
        base = {"row_id": r.row_id, "subset": r.subset, "proto_name": r.proto_name, "formula_raw": r.formula_raw,
                "Value": r.Value, "dataset_temperature_K": r.temperature, "dataset_C_rate": r.current_rate,
                "dataset_cycle": r.cycle_value, "trace_tier": r.tier, "DOI": r.DOI}
        if r.tier not in SINGLE or r.DOI not in CURATED:
            rows.append({**base, "source_pdf": "", "status": "來源未確定（09 未追到單一論文），未抽取",
                         **{k: "未抽取" for k in ["V_low", "V_high", "voltage_source", "electrolyte", "current",
                                                 "C_definition", "temperature", "activation", "cycle", "cycle_source"]}})
            continue
        c = CURATED[r.DOI]
        low, high, vsrc = c["voltage"]
        if "voltage_rate_test" in c and float(r.current_rate) != 0.2:
            low, high, vsrc = c["voltage_rate_test"]
        snips = [s for s in cands[r.DOI]["found"]["capacity_values"] if s["value"] == f"{r.Value:g}"]
        cyc, cyc_src = cycle_from_snippets(snips)
        if not cyc.startswith(("未找到", "數值未在")):
            cyc = "未找到（自動判讀，待人工確認）"  # only hand-checked cycle numbers are reported
        cyc, cyc_src = MANUAL_CYCLE.get((r.DOI, f"{r.Value:g}"), (cyc, cyc_src))
        rows.append({**base, "source_pdf": cands[r.DOI]["pdf"], "status": "已抽取",
                     "V_low": low, "V_high": high, "voltage_source": vsrc, "electrolyte": c["electrolyte"],
                     "current": c["current"], "C_definition": c["c_def"], "temperature": c["temperature"],
                     "activation": c["activation"], "cycle": cyc, "cycle_source": cyc_src})
    table = pd.DataFrame(rows)
    table.to_csv(LIT / "test_conditions.csv", index=False, encoding="utf-8-sig")
    table.to_csv(OUT / "14_test_conditions.csv", index=False, encoding="utf-8-sig")
    return table


# ----------------------------------------------------------------------------- sensitivity retrains
def sensitivity(table: pd.DataFrame) -> dict:
    import numpy as np
    from catboost import CatBoostRegressor
    from pymatgen.core import Composition

    import candidate_checks as t
    from composition_counterparts import TABLE3, counterparts
    from literature_pairs import features
    from rebuild_data_and_models import CB_PARAMS

    models_dir = t.HERE.parent / "models_local" / "upper_voltage_sensitivity_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    features_all, y, raw = t.training_table()
    evaluator = t.ssb.CandidateEvaluator()
    entries = []
    for formula, _, _ in TABLE3:
        entries.append((formula, "candidate", formula))
        entries += [(formula, label, other) for label, other in counterparts(formula)]
    entries.append(("Li2MnO3", "reference", "Li2MnO3"))
    formulas = [Composition(e[2]).formula.replace(" ", "") for e in entries]
    X = features(evaluator, formulas, 0.5, 1.0)
    saved = np.asarray(evaluator.model.predict(X, thread_count=1), dtype=float)

    def high(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    t2 = table.copy()
    t2["V_high_num"] = t2["V_high"].map(high)
    ambiguous = t2["V_high"].astype(str).str.contains("或")
    t2.loc[ambiguous, "V_high_num"] = 4.6  # "4.6 或 4.7": both values are >= 4.6 and < 4.8
    scenarios = {"exclude_Vhigh_ge_4.8": t2[t2["V_high_num"] >= 4.8]["row_id"].astype(int).tolist(),
                 "exclude_Vhigh_lt_4.6": t2[t2["V_high_num"] < 4.6]["row_id"].astype(int).tolist()}
    rows, info = [], {}
    for name, drop in scenarios.items():
        mask = ~raw["row_id"].isin(drop).to_numpy()
        model = CatBoostRegressor(**CB_PARAMS)
        model.fit(features_all.loc[mask], y.loc[mask])
        model.save_model(str(models_dir / f"{name}.cbm"))
        pred = np.asarray(model.predict(X, thread_count=1), dtype=float)
        info[name] = {"n_dropped": len(drop), "n_train": int(mask.sum()),
                      "dropped": t2[t2["row_id"].isin(drop)][["row_id", "formula_raw", "Value", "V_high", "DOI"]]
                      .to_dict(orient="records")}
        for (parent, label, _), f, s, p in zip(entries, formulas, saved, pred):
            rows.append({"scenario": name, "parent_candidate": parent, "variant": label, "Formula": f,
                         "pred_saved_model": s, "pred_retrained": p, "change": p - s})
        log(f"{name}: dropped {len(drop)}")
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "14_sensitivity_predictions.csv", index=False, encoding="utf-8-sig")
    ranks = {}
    for name, part in out[out["variant"].eq("candidate")].groupby("scenario"):
        ranks[name] = {"saved_order": part.sort_values("pred_saved_model", ascending=False)["Formula"].tolist(),
                       "retrained_order": part.sort_values("pred_retrained", ascending=False)["Formula"].tolist()}
    return {"scenarios": info, "candidate_rank": ranks}


if __name__ == "__main__":
    cands = extract()
    table = build(cands)
    write_json("14_summary.json", sensitivity(table))
