"""Source audit and capacity definitions (analysis 09; see revision_analyses/README.md).

Read-only on <BI> = the 2025 data clean-up folder (not distributed; set BDE_PIPELINE_DIR).

Provenance chain recovered here:
  step21_verfication1000/final.csv (1,008 rows; identical to paper1/06_raw_data/final_1008rows_with_anodes.csv)
    <- step20_structure_doping/final_origin.csv rows with done? == 'y' (1,011), edited in place during the
       manual verification, so rows are paired by position inside difflib equal / equal-length blocks;
    <- the DOI-bearing tables of the BatteryDataExtractor clean-up (Step14 ... step18), which keep the BDE
       fields Name, Specifier, Info, Property, Unit and DOI.  Rows are matched by (capacity value, C-rate,
       cycle) from the final row or from its final_origin row, and the composition is compared.
clean_rows.csv row_id i is the i-th cathode row of the 1,008-row table (checked: formula and Value equal).

Outputs: outputs/09_*, literature/records_to_check.csv.
"""
from __future__ import annotations

import ast
import difflib
import os
import re

import numpy as np
import pandas as pd
from pymatgen.core import Composition

import candidate_checks as t
from candidate_checks import OUT, SAVED, SITE_LIMIT, log, write_json

BI = t.Path(os.environ.get("BDE_PIPELINE_DIR", "BDE_PIPELINE_DIR_not_set"))  # 2025 clean-up tables, not distributed
LIT = t.HERE.parent / "literature"
DOI_TABLES = {
    "step18": "step18_add_doping/electrode_data_with_doping.csv",
    "step17_grouped": "step17_remove_duplicates/electrode_data_grouped_by_doi.csv",
    "step16": "step16_separate_type/electrode_data.csv",
    "step15": "Step15_add_type_and_correction/try_dopant.csv",
    "step14": "Step14_add_source/filtered_single_group_with_ref_doi_sorted_cleaned.csv",
}
TEXT_COLUMNS = ["Property", "Name", "Unit", "Specifier", "Type", "Info"]
NOMINAL_ANIONS = {"Li2MnO3": 3.0, "Li-rich-NMC": 2.0, "NMC-532": 2.0, "NMC-622": 2.0, "NMC-811": 2.0, "NCA": 2.0,
                  "LiNiO2": 2.0, "LiCoO2": 2.0, "LiMn2O4": 4.0, "LNMO-spinel": 4.0, "LiFePO4": 4.0, "LiMnPO4": 4.0}
KEYWORDS = ["discharge", "charge", "initial", "first", "reversible", "specific", "theoretical", "capacity"]


def frac(formula: str) -> dict[str, float] | None:
    try:
        return Composition(formula).fractional_composition.get_el_amt_dict()
    except Exception:  # noqa: BLE001 - BDE names such as "Li2S / rGO" do not parse
        return None


def extracted_frac(text: str) -> dict[str, float] | None:
    try:
        items = ast.literal_eval(str(text))
        comp = Composition({k: float(v) for k, v in items[0].items()})
        return comp.fractional_composition.get_el_amt_dict()
    except Exception:  # noqa: BLE001
        return None


def same_comp(a: dict | None, b: dict | None, tol: float = 0.002) -> bool:
    if not a or not b:
        return False
    keys = set(a) | set(b)
    return max(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys) <= tol


def key(value, rate, cycle) -> tuple:
    return (round(float(value), 2), round(float(rate), 4), str(cycle).strip())


DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"<>,;]+", re.I)
PDF_DIRS = {"Paper": BI / "Paper", "named_20260102": BI.parent / "20260102", "Verification": BI.parent.parent / "Verification"}


def pdf_dois() -> pd.DataFrame:
    """DOI printed on the first page(s) of every PDF kept from the 2025 verification (cached)."""
    import subprocess

    cache = OUT / "09_pdf_dois.csv"
    if cache.exists():
        return pd.read_csv(cache, encoding="utf-8-sig")
    rows = []
    for kind, folder in PDF_DIRS.items():
        for pdf in sorted(folder.rglob("*.pdf")):
            if not pdf.is_file():
                continue
            doi = None
            for last in (1, 2):
                text = subprocess.run(["pdftotext", "-l", str(last), "-q", str(pdf), "-"], capture_output=True,
                                      text=True, encoding="utf-8", errors="replace").stdout
                hits = [h.rstrip(".)]") for h in DOI_RE.findall(text)]
                if hits:
                    doi = max(set(hits), key=lambda h: (hits.count(h), -hits.index(h)))
                    break
            stem = pdf.stem
            m = re.match(r"^(.*?)\s+([\d.]+)$", stem)
            rows.append({"kind": kind, "file": str(pdf.relative_to(BI.parent.parent)), "stem": stem,
                         "name_formula": m.group(1) if (m and kind == "named_20260102") else None,
                         "name_value": float(m.group(2)) if (m and kind == "named_20260102") else None,
                         "DOI": doi})
    table = pd.DataFrame(rows)
    table.to_csv(cache, index=False, encoding="utf-8-sig")
    return table


def position_map(origin_y: pd.DataFrame, final: pd.DataFrame) -> dict[int, tuple[int, str]]:
    """final row -> (final_origin row, how it was paired)."""
    a = list(zip(origin_y["Name"].astype(str).str.strip(), origin_y["Value"].astype(float).round(3),
                 origin_y["current_rate"].astype(float).round(4), origin_y["cycle_value"].astype(str)))
    b = list(zip(final["formula"].astype(str).str.strip(), final["Value"].astype(float).round(3),
                 final["current_rate"].astype(float).round(4), final["cycle_value"].astype(str)))
    mapping = {}
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            for k in range(j2 - j1):
                mapping[j1 + k] = (int(origin_y["origin_row"].iloc[i1 + k]), "identical")
        elif tag == "replace" and (i2 - i1) == (j2 - j1):
            for k in range(j2 - j1):
                mapping[j1 + k] = (int(origin_y["origin_row"].iloc[i1 + k]), "same position, edited")
    return mapping


def trace() -> pd.DataFrame:
    final = pd.read_csv(BI / "step21_verfication1000" / "final.csv", low_memory=False)
    origin = pd.read_csv(BI / "step20_structure_doping" / "final_origin.csv", low_memory=False)
    origin["origin_row"] = np.arange(len(origin))
    origin_y = origin[origin["done?"].eq("y")].reset_index(drop=True)
    mapping = position_map(origin_y, final)

    pool = []
    for tag, rel in DOI_TABLES.items():
        d = pd.read_csv(BI / rel, low_memory=False)
        d["table"] = tag
        pool.append(d)
    pool = pd.concat(pool, ignore_index=True)
    pool = pool[pd.to_numeric(pool["Value"], errors="coerce").notna()].copy()
    pool["k"] = [key(v, c, y) for v, c, y in zip(pool["Value"], pool["current_value"], pool["cycle_value"])]
    pool["comp"] = [extracted_frac(x) or frac(str(n).split("/")[0].split("@")[0].strip())
                    for x, n in zip(pool["Extracted_name"], pool["Name"])]
    by_key = pool.groupby("k").indices
    pdfs = pdf_dois()
    pdf_doi_set = {str(d).lower() for d in pdfs["DOI"].dropna()}
    named = pdfs[pdfs["kind"].eq("named_20260102") & pdfs["DOI"].notna()].copy()
    named["comp"] = [frac(re.sub(r"[@&].*$", "", str(f)).strip()) for f in named["name_formula"]]

    def split_dois(series: pd.Series) -> list[str]:
        return sorted({d.strip() for s in series.dropna().astype(str) for d in s.split(",") if d.strip()},
                      key=str.lower)

    rows = []
    for i, r in final.iterrows():
        target = frac(str(r["formula"]))
        origin_row, how = mapping.get(i, (None, "not paired"))
        keys = {key(r["Value"], r["current_rate"], r["cycle_value"])}
        if origin_row is not None:
            o = origin.iloc[origin_row]
            keys.add(key(o["Value"], o["current_rate"], o["cycle_value"]))
        idx = sorted({j for k in keys for j in by_key.get(k, [])})
        cands = pool.iloc[idx]
        comp_ok = cands[[same_comp(target, c) for c in cands["comp"]]]
        chosen = comp_ok if len(comp_ok) else cands
        dois = split_dois(chosen["DOI"])
        named_hit = named[[same_comp(target, c) for c in named["comp"]]]
        named_value = named_hit[np.isclose(named_hit["name_value"].astype(float), float(r["Value"]))]
        comp_only = pool[[same_comp(target, c) for c in pool["comp"]]] if target else pool.iloc[0:0]
        comp_dois = split_dois(comp_only["DOI"])
        comp_dois_in_pdf = [d for d in comp_dois if d.lower() in pdf_doi_set]
        if len(comp_ok) and len(dois) == 1:
            status = "A 唯一 DOI（數值、C-rate、cycle、組成相符）"
        elif len(comp_ok):
            status = "A' 多個 DOI（數值、C-rate、cycle、組成相符）"
        elif len(named_value):
            status, dois = "B 以「化學式 數值」命名的驗證 PDF（組成、數值相符）", sorted(set(named_value["DOI"]))
        elif len(comp_dois_in_pdf) == 1:
            status, dois = "C 組成相符，且候選中只有一篇在驗證 PDF 中", comp_dois_in_pdf
        elif len(comp_dois_in_pdf) > 1:
            status, dois = "C' 組成相符，候選中有多篇在驗證 PDF 中", comp_dois_in_pdf
        elif comp_dois:
            status, dois = "D 只有組成相符（候選論文不在驗證 PDF 中）", comp_dois
        elif len(cands):
            status = "E 數值、C-rate、cycle 相符，但組成不符"
        else:
            status = "F 追不到"
        best = chosen.iloc[0] if len(chosen) else (comp_only.iloc[0] if len(comp_only) else None)
        rows.append({
            "row_1008": i, "formula_raw": r["formula"], "Value": r["Value"], "Type": r["Type"],
            "temperature": r["temperature"], "current_rate": r["current_rate"], "cycle_value": r["cycle_value"],
            "composite": r["composite"], "Extracted_name": r["Extracted_name"],
            "final_origin_row": origin_row, "final_origin_pairing": how,
            "final_origin_name": origin.iloc[origin_row]["Name"] if origin_row is not None else None,
            "trace_status": status, "DOI": "; ".join(dois), "n_DOI": len(dois),
            **{f"BDE_{c}": (best[c] if best is not None and c in best and pd.notna(best[c]) else None)
               for c in TEXT_COLUMNS},
            "BDE_specifiers_all": (" | ".join(sorted(set(chosen["Specifier"].dropna().astype(str)))) if len(chosen)
                                   else None),
            "BDE_specifiers_same_composition": (" | ".join(sorted(set(comp_only["Specifier"].dropna().astype(str))))
                                                if len(comp_only) else None),
            "n_candidate_DOI_same_composition": len(comp_dois),
        })
    out = pd.DataFrame(rows)
    # Manual check (2026-10-23): rows 824-838 are one contiguous block of final_origin (rows 1492-1506).
    # The Dong 2015 PDF kept in Verification/285.pdf (DOI 10.1039/C4TA02924K, ref. [64]) lists
    # "Li2MnO2.9795, Li2MnO2.9390F0.015, Li2MnO2.9145F0.030, Li2MnO2.9030F0.045" with their oxygen-vacancy
    # values, and 13 of the 15 capacities appear verbatim in its text (168.4 = 183 x 92 % retention;
    # 144 not found in the text).
    dong = out["row_1008"].between(824, 838) & ~out["trace_status"].str.startswith("A")
    out.loc[dong, "trace_status"] = "G 人工核對：化學式與容量見於來源 PDF（Verification/285.pdf）"
    out.loc[dong, "DOI"] = "10.1039/C4TA02924K"
    out.loc[dong, "n_DOI"] = 1
    out.to_csv(OUT / "09_trace_1008.csv", index=False, encoding="utf-8-sig")
    return out


def capacity_classes(spec: str | None) -> dict[str, bool]:
    text = (spec or "").lower()
    flags = {k: bool(re.search(rf"\b{k}\b", text)) for k in KEYWORDS}
    flags["charge"] = bool(re.search(r"(?<!dis)charge", text))
    return flags


def main() -> None:
    log("trace")
    trace_table = trace()
    clean = pd.read_csv(SAVED / "data" / "clean_rows.csv").sort_values("row_id").reset_index(drop=True)
    cathode = trace_table[trace_table["Type"].astype(str).str.contains("Cathode")].reset_index(drop=True)
    if not (cathode["formula_raw"].values == clean["formula"].values).all():
        raise RuntimeError("cathode rows of the 1,008-row table do not line up with clean_rows")
    rec = pd.concat([clean[["row_id", "canonical_key", "canonical_formula"]], cathode], axis=1)
    assign = pd.read_csv(SAVED / "results" / "known_control_prototype_assignment.csv").set_index("canonical_key")
    rec["proto_name"] = rec["canonical_key"].map(assign["proto_name"])

    # capacity definitions for all 907 records: the Specifier of the exact-match BDE rows (tier A/A')
    spec_flags = pd.DataFrame([capacity_classes(s) for s in rec["BDE_specifiers_all"]])
    rec = pd.concat([rec, spec_flags.add_prefix("spec_")], axis=1)
    rec.to_csv(OUT / "09_records_907.csv", index=False, encoding="utf-8-sig")

    # records to check: Li2MnO3, Li-rich-NMC templates and every Cr / F composition
    def elements(f: str) -> set[str]:
        return set(Composition(f).get_el_amt_dict())

    has_crf = rec["canonical_formula"].map(lambda f: bool({"Cr", "F"} & elements(f)))
    sel = rec[rec["proto_name"].isin(["Li2MnO3", "Li-rich-NMC"]) | has_crf].copy()
    sel["subset"] = np.where(sel["proto_name"].isin(["Li2MnO3", "Li-rich-NMC"]), sel["proto_name"], "含 Cr 或 F（無模板或其他模板）")
    flags = []
    for r in sel.itertuples(index=False):
        notes = []
        raw_comp, canon = Composition(str(r.formula_raw)), Composition(str(r.canonical_formula))
        amounts = canon.get_el_amt_dict()
        anions = amounts.get("O", 0) + amounts.get("F", 0)
        cations = sum(v for e, v in amounts.items() if e not in ("O", "F"))
        ratio = cations / anions
        nominal = NOMINAL_ANIONS.get(r.proto_name)
        if nominal and abs(anions - nominal) > 0.01:
            notes.append(f"陰離子總數 {anions:g}，模板名義值 {nominal:g}")
        cls = t.STRUCTURAL_FAMILY.get(r.proto_name) if isinstance(r.proto_name, str) else None
        if cls and ratio > SITE_LIMIT[cls] + 1e-9:
            notes.append(f"陽/陰 {ratio:.4f} 超過 {cls} 上限 {SITE_LIMIT[cls]:g}")
        if not same_comp(raw_comp.fractional_composition.get_el_amt_dict(),
                         canon.fractional_composition.get_el_amt_dict(), 1e-6):
            notes.append("原始字串與 canonical 組成不同")
        if r.final_origin_name is not None and frac(str(r.final_origin_name)) and \
                not same_comp(frac(str(r.final_origin_name)), raw_comp.fractional_composition.get_el_amt_dict(), 1e-3):
            notes.append(f"驗證前名稱為 {r.final_origin_name}")
        flags.append({"cation_to_anion": ratio, "anion_total": anions, "flag": "；".join(notes)})
    sel = pd.concat([sel.reset_index(drop=True), pd.DataFrame(flags)], axis=1)
    cols = ["subset", "proto_name", "row_id", "row_1008", "formula_raw", "canonical_formula", "Value", "temperature",
            "current_rate", "cycle_value", "Type", "composite", "Extracted_name", "BDE_Name", "BDE_Property",
            "BDE_Unit", "BDE_Specifier", "BDE_Info", "BDE_specifiers_all", "final_origin_name",
            "final_origin_pairing", "trace_status", "n_DOI", "DOI", "cation_to_anion", "anion_total", "flag"]
    sel = sel.sort_values(["subset", "canonical_formula", "Value"])[cols]
    sel.to_csv(LIT / "records_to_check.csv", index=False, encoding="utf-8-sig")
    sel.to_csv(OUT / "09_records_to_check.csv", index=False, encoding="utf-8-sig")

    def counts(frame: pd.DataFrame) -> dict[str, int]:
        return {k: int(frame[f"spec_{k}"].sum()) for k in KEYWORDS}

    write_json("09_summary.json", {
        "trace_status_1008": trace_table["trace_status"].value_counts().to_dict(),
        "trace_status_907": rec["trace_status"].value_counts().to_dict(),
        "pairing_1008": trace_table["final_origin_pairing"].value_counts().to_dict(),
        "unique_dois_907": len({d for s in rec["DOI"] for d in str(s).split("; ") if d}),
        "records_to_check": int(len(sel)), "flagged": int(sel["flag"].ne("").sum()),
        "keyword_counts_907": counts(rec),
        "keyword_counts_Li2MnO3": counts(rec[rec["proto_name"].eq("Li2MnO3")]),
        "keyword_counts_LiRichNMC": counts(rec[rec["proto_name"].eq("Li-rich-NMC")]),
        "no_specifier_907": int(rec["BDE_specifiers_all"].isna().sum()),
    })


if __name__ == "__main__":
    main()
