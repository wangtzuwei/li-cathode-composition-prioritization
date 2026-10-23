"""Literature processing for analysis 11.

  copies     versioned copies of every CSV in literature (originals untouched)
  checklist  literature/pdfs/DOWNLOAD_CHECKLIST.md from the prioritized download list
  crossref   query every DOI in the literature CSVs on the Crossref API, compare
             author / year / title, write candidate_refs.bib (BibTeX from Crossref)
  external   extend the 06b item-1 external G3 comparison with Task B / Task C materials

Only GET requests to api.crossref.org are sent, carrying nothing but the DOI.
Usage:  python literature_11.py [copies checklist crossref external]   (default: all)
"""
from __future__ import annotations

import difflib
import json
import re
import shutil
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from fractions import Fraction
from pathlib import Path

import pandas as pd

import followup_checks as f6
from candidate_checks import OUT, log, write_json

LIT = f6.t.HERE.parent / "literature"
PDFS = LIT / "pdfs"
VERSION = "_v1_20261023"
SOURCE_FILES = [
    "Cr_F_LiRich_primary_verified_samples.csv",
    "Cr_F_LiRich_excluded_screening.csv",
    "Task_B_verified_synthesis_evidence.csv",
    "Task_B_fulltext_and_SI_pending.csv",
    "Task_C_degradation_mechanisms_verified.csv",
    "Tasks_ABC_prioritized_paper_downloads.csv",
]
USER_AGENT = "paper1-literature-check/1.0 (python urllib)"


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(LIT / name, encoding="utf-8-sig", dtype=str).fillna("")


# ----------------------------------------------------------------------------- copies
def copies() -> None:
    log("copies")
    for name in SOURCE_FILES:
        src = LIT / name
        dst = LIT / f"{src.stem}{VERSION}{src.suffix}"
        if dst.exists():
            log(f"  exists, not overwritten: {dst.name}")
            continue
        shutil.copy2(src, dst)
        log(f"  {name} -> {dst.name}")


# ----------------------------------------------------------------------------- checklist
def doi_filename(doi: str) -> str:
    return doi.strip().replace("/", "_")


def checklist() -> None:
    log("checklist")
    PDFS.mkdir(exist_ok=True)
    table = read("Tasks_ABC_prioritized_paper_downloads.csv")
    table["order"] = table["下載優先度"].map({"P0": 0, "P1": 1, "P2": 2})
    table = table.sort_values(["order"], kind="mergesort")
    raw_path = OUT / "11_crossref_raw.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.exists() else {}

    def crossref_label(doi: str) -> str:
        msg = raw.get(doi)
        if not msg:
            return "（Crossref 查不到）" if doi in raw else "（尚未查詢）"
        first = (msg.get("author") or [{}])[0].get("family", "")
        year = ((msg.get("issued") or {}).get("date-parts") or [[""]])[0][0]
        journal = (msg.get("container-title") or [""])[0].replace("&amp;", "&")
        return f"{first} {year}, *{journal}*"
    lines = [
        "# 論文下載清單",
        "",
        "依 `../Tasks_ABC_prioritized_paper_downloads.csv`（GPT 整理）產生，依優先度 P0 → P1 → P2 排序，同一優先度內保留原檔順序。",
        "",
        "- 建議檔名：DOI 中的 `/` 換成 `_`。主文存成 `<檔名>.pdf`，SI 存成 `<檔名>_SI.pdf`；SI 有多個檔案時，依序加上 `_SI1`、`_SI2`。",
        "- 下載後放在這個資料夾（`literature/pdfs/`）。",
        "- 「要下載的內容」中寫「若有」的，表示原清單不確定該論文是否有 SI。",
        "- 「Crossref」欄是 2026-10-23 從 Crossref 查到的第一作者、年份（issued）和期刊，供下載時對照。",
        "",
    ]
    for priority in ["P0", "P1", "P2"]:
        part = table[table["下載優先度"].eq(priority)]
        lines += [f"## {priority}（{len(part)} 篇）", "",
                  "| 完成 | DOI | 研究／作者（原清單） | Crossref | 支援任務 | 要下載的內容 | 主文檔名 | SI 檔名 |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in part.itertuples(index=False):
            content = r[4]
            wants_main = ("全文" in content) or ("主文" in content)
            wants_si = "Supporting" in content or "SI" in content
            stem = doi_filename(r[1])
            main = f"`{stem}.pdf`" if wants_main else "—"
            si = f"`{stem}_SI.pdf`" if wants_si else "—"
            lines.append(f"| ☐ | [{r[1]}](https://doi.org/{r[1]}) | {r[2]} | {crossref_label(r[1])} | {r[3]} | "
                         f"{content} | {main} | {si} |")
        lines.append("")
    (PDFS / "DOWNLOAD_CHECKLIST.md").write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------------- crossref
def collect_dois() -> pd.DataFrame:
    rows = []
    specs = {
        "Cr_F_LiRich_primary_verified_samples.csv": ("DOI", "作者_簡稱", "年份", "論文標題"),
        "Cr_F_LiRich_excluded_screening.csv": ("DOI", "論文或關鍵詞", "年份", None),
        "Task_B_verified_synthesis_evidence.csv": ("DOI", "第一作者", "年份", None),
        "Task_B_fulltext_and_SI_pending.csv": ("DOI", "研究", None, None),
        "Task_C_degradation_mechanisms_verified.csv": ("DOI", "作者_年份", None, None),
        "Tasks_ABC_prioritized_paper_downloads.csv": ("DOI", "研究/作者", None, None),
    }
    for name, (doi_col, author_col, year_col, title_col) in specs.items():
        table = read(name)
        for i, r in table.iterrows():
            doi = r[doi_col].strip()
            if not doi:
                continue
            author_text = r[author_col]
            year_text = r[year_col] if year_col else author_text
            year = re.search(r"(19|20)\d{2}", year_text)
            title = r[title_col] if title_col else ""
            if name == "Cr_F_LiRich_excluded_screening.csv":
                # "Kim et al. Effects of ..." -> author part and title fragment
                m = re.match(r"^(.*?et al\.)\s*(.*)$", author_text)
                title = m.group(2) if m else ""
            rows.append({"file": name, "row": i + 2, "DOI": doi, "csv_author": author_text,
                         "csv_year": year.group(0) if year else "", "csv_title": title})
    return pd.DataFrame(rows)


def fetch(url: str) -> bytes | None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            time.sleep(2 * (attempt + 1))
        except Exception:  # noqa: BLE001 - network hiccup, retry
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Crossref request failed: {url}")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"<[^>]+>", "", text)  # MathML / HTML tags in Crossref titles
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def surname_candidates(author_text: str, file: str) -> list[str]:
    """Author names written in the CSV cell; [] when the cell only describes the topic."""
    if "et al" in author_text:
        text = author_text.split("et al")[0]
    elif file == "Task_B_verified_synthesis_evidence.csv":
        text = author_text  # the column holds the first author's name
    else:
        match = re.match(r"^([A-Z][^\s\d\-/]+)\s+(19|20)\d{2}", author_text)  # "Fan 2024"
        text = match.group(1) if match else ""
    return [normalize(token) for token in re.split(r"[\s,]+", text) if len(normalize(token)) >= 2]


def crossref() -> None:
    log("crossref")
    refs = collect_dois()
    unique = sorted(set(refs["DOI"]), key=str.lower)
    raw_path, bib_path = OUT / "11_crossref_raw.json", OUT / "11_crossref_bibtex.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.exists() else {}
    bib = json.loads(bib_path.read_text(encoding="utf-8")) if bib_path.exists() else {}
    for doi in unique:
        if doi in raw and (raw[doi] is None or doi in bib):
            continue  # cached from an earlier run
        data = fetch("https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="/"))
        raw[doi] = json.loads(data)["message"] if data else None
        if data:
            bib_bytes = fetch("https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="/")
                              + "/transform/application/x-bibtex")
            bib[doi] = bib_bytes.decode("utf-8").strip() if bib_bytes else None
        time.sleep(0.3)
        log(f"  {doi}: {'found' if raw[doi] else 'NOT FOUND'}")
    raw_path.write_text(json.dumps(raw, indent=1, ensure_ascii=False), encoding="utf-8")
    bib_path.write_text(json.dumps(bib, indent=1, ensure_ascii=False), encoding="utf-8")

    checks = []
    for r in refs.itertuples(index=False):
        msg = raw.get(r.DOI)
        if msg is None:
            checks.append({**r._asdict(), "found": False})
            continue
        families = [normalize(a.get("family", "") or a.get("name", "")) for a in msg.get("author", [])]
        first = families[0] if families else ""
        years = {}
        for key in ["issued", "published-print", "published-online", "published"]:
            parts = (msg.get(key) or {}).get("date-parts") or [[None]]
            if parts[0] and parts[0][0]:
                years[key] = str(parts[0][0])
        cr_title = (msg.get("title") or [""])[0]
        cands = surname_candidates(r.csv_author, r.file)
        if not cands:
            author_status = "CSV 未寫作者（只有主題描述）"
        elif first and any(c == first for c in cands):
            author_status = "一致（第一作者）"
        elif any(c in families for c in cands):
            author_status = "不符：CSV 作者不是 Crossref 第一作者"
        else:
            author_status = "不符"
        if not r.csv_year:
            year_status = "CSV 未寫年份"
        elif r.csv_year == years.get("issued"):
            year_status = "一致"
        elif r.csv_year in years.values():
            year_status = "部分一致（符合 " + ",".join(k for k, v in years.items() if v == r.csv_year) + "）"
        else:
            year_status = "不符"
        if r.file == "Cr_F_LiRich_excluded_screening.csv":
            # this column is a short description, not a title: compared by hand in the report
            title_status, ratio = "CSV 為簡述，人工比對", None
        elif not r.csv_title:
            title_status, ratio = "CSV 未寫標題", None
        else:
            a, b = normalize(r.csv_title), normalize(cr_title)
            ratio = difflib.SequenceMatcher(None, a, b).ratio()
            title_status = "一致" if (ratio >= 0.95 or (len(a) >= 20 and a in b)) else "不符"
        checks.append({**r._asdict(), "found": True,
                       "crossref_first_author": (msg.get("author") or [{}])[0].get("family", ""),
                       "crossref_authors_n": len(families),
                       "crossref_year_issued": years.get("issued", ""),
                       "crossref_years_all": json.dumps(years),
                       "crossref_title": cr_title,
                       "crossref_container": (msg.get("container-title") or [""])[0],
                       "author_status": author_status, "year_status": year_status,
                       "title_status": title_status, "title_ratio": ratio})
    checks = pd.DataFrame(checks)
    checks.to_csv(OUT / "11_doi_check.csv", index=False, encoding="utf-8-sig")

    header = ["% Candidate references from literature/*.csv (GPT-curated), BibTeX as returned by",
              "% the Crossref API on 2026-10-23. Candidate list only; the manuscript's refs.bib is untouched.",
              "% DOIs not found on Crossref are listed in analysis 11.", ""]
    entries = [bib[d] for d in unique if bib.get(d)]
    (LIT / "candidate_refs.bib").write_text("\n".join(header) + "\n\n".join(entries) + "\n", encoding="utf-8")
    write_json("11_crossref_summary.json", {
        "unique_dois": len(unique), "found": sum(v is not None for v in raw.values()),
        "not_found": [d for d, v in raw.items() if v is None],
        "bibtex_entries": len(entries),
        "bibtex_missing": [d for d in unique if raw.get(d) and not bib.get(d)],
    })


# ----------------------------------------------------------------------------- external
def fmt(value: Fraction | float) -> str:
    return f"{float(value):.6g}"


def task_bc_materials() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Materials with a complete nominal formula in Task B / Task C (formulas typed from the CSV cells)."""
    rows: list[dict[str, str]] = []
    # Task B row 1 - Zhang 2013: Li(1+2x) Mn(0.3+x) Ni(0.3-3x) Cr0.4 O2, x = 1/60 ... 1/12 (samples A-E)
    for label, x in zip("ABCDE", [Fraction(1, 60), Fraction(1, 30), Fraction(1, 20), Fraction(1, 15), Fraction(1, 12)]):
        rows.append({"source": "Zhang 2013", "DOI": "10.3724/SP.J.1077.2013.12412", "file": "Task B",
                     "sample": f"{label}（x = {x}）", "nominal": "Li_(1+2x)Mn_(0.3+x)Ni_(0.3−3x)Cr_0.4O_2",
                     "Formula": f"Li{fmt(1 + 2 * x)}Mn{fmt(Fraction(3, 10) + x)}Ni{fmt(Fraction(3, 10) - 3 * x)}Cr0.4O2",
                     "structure": "Li-rich layered"})
    rows.append({"source": "Fan 2024", "DOI": "10.1021/acsenergylett.3c02509", "file": "Task B", "sample": "LNMO",
                 "nominal": "Li1.2Ni0.2Mn0.6O2", "Formula": "Li1.2Ni0.2Mn0.6O2", "structure": "Li-rich layered"})
    for x in ["0", "0.01", "0.03", "0.05"]:
        xv = float(x)
        rows.append({"source": "Ming 2018", "DOI": "10.3389/fchem.2018.00076", "file": "Task B; Task C",
                     "sample": f"x = {x}", "nominal": "Li1.2Mn0.54−xNbxCo0.13Ni0.13O2−6xF6x",
                     "Formula": (f"Li1.2Mn{0.54 - xv:.6g}Co0.13Ni0.13O{2 - 6 * xv:.6g}"
                                 + (f"Nb{xv:g}F{6 * xv:.6g}" if xv else "")),
                     "structure": "Li-rich layered（Nb/F 共摻）"})
    for formula, sample in [("Li1.25Mn0.5Ti0.3O1.95F0.05", "DRX 低 F"), ("Li1.35Mn0.7Ti0.1O1.85F0.15", "DRX F0.15")]:
        rows.append({"source": "Chambers 2025", "DOI": "10.1039/D4MA01287A", "file": "Task B", "sample": sample,
                     "nominal": formula, "Formula": formula, "structure": "DRX（非層狀）"})
    for formula, sample in [("Li2MnO2F", "LMOF"), ("Li2Mn0.5V0.5O2F", "LMVOF")]:
        rows.append({"source": "Blumenhofer 2023", "DOI": "10.1021/acsmaterialsau.2c00064", "file": "Task B",
                     "sample": sample, "nominal": formula, "Formula": formula, "structure": "DRX（非層狀）"})
    rows += [
        {"source": "Marie 2024", "DOI": "10.1038/s41563-024-01833-z", "file": "Task C", "sample": "",
         "nominal": "Li1.2Ni0.13Co0.13Mn0.54O2", "Formula": "Li1.2Ni0.13Co0.13Mn0.54O2", "structure": "Li-rich layered"},
        {"source": "Hua 2019", "DOI": "10.1038/s41467-019-13240-z", "file": "Task C", "sample": "Li[Li0.2Ni0.2Mn0.6]O2",
         "nominal": "Li[Li0.2Ni0.2Mn0.6]O2", "Formula": "Li1.2Ni0.2Mn0.6O2", "structure": "Li-rich layered"},
        {"source": "Nakayama 2020", "DOI": "10.1038/s41467-020-18285-z", "file": "Task C", "sample": "單晶",
         "nominal": "Li2MnO3", "Formula": "Li2MnO3", "structure": "Li2MnO3"},
        {"source": "Ammundsen 2002", "DOI": "10.1149/1.1456535", "file": "Task C", "sample": "",
         "nominal": "Li1.2Cr0.4Mn0.4O2", "Formula": "Li1.2Cr0.4Mn0.4O2", "structure": "Li-rich layered"},
        {"source": "Moździerz 2025", "DOI": "10.1002/aenm.202502157", "file": "Task C", "sample": "LCT",
         "nominal": "Li1.2Cr0.4Ti0.4O2", "Formula": "Li1.2Cr0.4Ti0.4O2", "structure": "部分無序 Li-excess"},
    ]
    duplicates_of_06b = [
        {"source": "Dong 2015", "DOI": "10.1039/C4TA02924K", "file": "Task B; Task C",
         "reason": "已在 06b 第 1 項（[64]，x = 0、0.015、0.03、0.045）"},
        {"source": "Lee 2015", "DOI": "10.1039/C5TA01214G", "file": "Task C",
         "reason": "已在 06b 第 1 項（primary CSV，x = 0、0.05、0.1、0.2）"},
        {"source": "Wang 2024", "DOI": "10.1021/acsenergylett.3c02697", "file": "Task C",
         "reason": "已在 06b 第 1 項（primary CSV，LNMO、F1、F2.5、F5）"},
        {"source": "Kucuk 2026", "DOI": "10.3390/batteries12040126", "file": "Task C",
         "reason": "已在 06b 第 1 項（primary CSV，x = 0–0.10）"},
    ]
    skipped = [
        {"source": "Fan 2024", "DOI": "10.1021/acsenergylett.3c02509", "file": "Task B",
         "material": "Cr-doped Li1.2Ni0.2Mn0.6O2（CLNMO，Cr 3%／5%／7%）",
         "reason": "摻雜樣的完整名義式正文未報告（CSV 註明 SI 未取得）"},
        {"source": "Hua 2019", "DOI": "10.1038/s41467-019-13240-z", "file": "Task C",
         "material": "LixNi0.2Mn0.6Oy", "reason": "x、y 未給定"},
        {"source": "Myeong 2018", "DOI": "10.1038/s41467-018-05802-4", "file": "Task C",
         "material": "Li1+xNiyCozMn1−x−y−zO2（有序／失序對照）", "reason": "x、y、z 未給定"},
        {"source": "Moździerz 2025", "DOI": "10.1002/aenm.202502157", "file": "Task C",
         "material": "LCT-HE（多陽離子置換）", "reason": "CSV 未寫化學式"},
        {"source": "Huang 2021（[65]）", "DOI": "10.1007/s10853-021-05922-1", "file": "06b",
         "material": "Cr/F 共摻 Li1.2Ni0.13Co0.13Mn0.54O2", "reason": "仍查不到共摻樣品的化學式（同 06b）"},
    ]
    return rows, duplicates_of_06b + skipped


def external() -> None:
    log("external")
    new_rows, not_added = task_bc_materials()
    rows = []
    for m in new_rows:
        rows.append(f6.g3_row(m["source"], m["nominal"], m["Formula"],
                              {"DOI": m["DOI"], "sample": m["sample"], "from": m["file"], "structure": m["structure"]}))
    new = pd.DataFrame(rows)
    old = pd.read_csv(OUT / "06b1_external_g3.csv")
    old["from"] = "06b 第 1 項"
    old["structure"] = ""
    combined = pd.concat([old, new], ignore_index=True)
    combined["G3_used"] = combined["G3_assigned_template"].fillna(combined["G3_Li2MnO3_template"])
    combined = combined.sort_values(["G3_used", "source", "Formula"], ascending=[False, True, True],
                                    kind="mergesort").reset_index(drop=True)
    combined.to_csv(OUT / "11_external_g3_combined.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(not_added).to_csv(OUT / "11_external_not_added.csv", index=False, encoding="utf-8-sig")
    write_json("11_external_summary.json", {"n_new": len(new), "n_combined": len(combined),
                                           "not_added": not_added})


ITEMS = {"copies": copies, "checklist": checklist, "crossref": crossref, "external": external}

if __name__ == "__main__":
    for item in sys.argv[1:] or list(ITEMS):
        ITEMS[item]()
