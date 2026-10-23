#!/usr/bin/env python3
"""Query the Materials Project for the Table 4 entries and the Li-Mn-Cr-O-F neighbourhood.

Prepared and run on 2026-10-23 (see README.md and analysis 19).  Needs MP_API_KEY:

    set MP_API_KEY=...            (Windows cmd)   /   $env:MP_API_KEY="..."   (PowerShell)
    python mp_table4_query.py

The key is read from the environment and never written to disk.  Every run writes a new folder
outputs/mp_table4/<UTC timestamp>/ with:
    meta.json                         query time (UTC), database version, mp-api / pymatgen / emmet
                                      versions, the exact queries, and any errors
    table4_summary.json               full summary documents of the 4 Table 4 entries (all fields)
    table4_thermo.json                full thermo documents (every thermo type) of the 4 entries
    table4_robocrys.json              Robocrystallographer descriptions (source of structure-type labels)
    neighbourhood_summary.json        full summary documents for every sub-system of Li-Mn-Cr-O-F
    neighbourhood_table.csv           one row per neighbourhood entry (id, formula, space group,
                                      energy above hull, stability flags)
    table4_comparison.csv             manuscript Table 4 values next to the values returned now
Nothing else is modified.
"""
from __future__ import annotations

import csv
import datetime as dt
import importlib.metadata as md
import itertools
import json
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ELEMENTS = ["Li", "Mn", "Cr", "O", "F"]
# Manuscript Table 4 ("Database version v2025.09.25; accessed 26 July 2026")
TABLE4 = [
    {"material_id": "mp-18988", "formula": "Li2MnO3", "space_group": "C2/m", "label": "caswellsilverite-like", "e_above_hull": 0.000},
    {"material_id": "mp-775130", "formula": "Li2CrO3", "space_group": "C2/m", "label": "caswellsilverite-like", "e_above_hull": 0.052},
    {"material_id": "mp-850919", "formula": "Li2Cr2MnO6", "space_group": "C2/m", "label": "", "e_above_hull": 0.056},
    {"material_id": "mp-753906", "formula": "Li2MnO2F", "space_group": "C2/m", "label": "beta-polonium-derived", "e_above_hull": None},
]
MANUSCRIPT_DB_VERSION = "v2025.09.25"


def dump(doc) -> dict:
    """pydantic document -> JSON-safe dict (all fields that the API returned)."""
    if hasattr(doc, "model_dump"):
        return doc.model_dump(mode="json")
    if hasattr(doc, "dict"):
        return json.loads(json.dumps(doc.dict(), default=str))
    return json.loads(json.dumps(doc, default=str))


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str), encoding="utf-8")


def versions() -> dict[str, str]:
    out = {"python": sys.version}
    for pkg in ["mp-api", "pymatgen", "emmet-core"]:
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def main() -> None:
    key = os.environ.get("MP_API_KEY", "").strip()
    if not key:
        sys.exit("MP_API_KEY is not set; nothing was queried.")
    from mp_api.client import MPRester

    started = dt.datetime.now(dt.timezone.utc)
    out = HERE.parent / "outputs" / "mp_table4" / started.strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=False)
    ids = [row["material_id"] for row in TABLE4]
    chemsys = ["-".join(sorted(c)) for r in range(1, len(ELEMENTS) + 1) for c in itertools.combinations(ELEMENTS, r)]
    meta = {"query_started_utc": started.isoformat(), "versions": versions(),
            "manuscript_database_version": MANUSCRIPT_DB_VERSION,
            "queries": {"table4_material_ids": ids, "neighbourhood_chemsys": chemsys}, "errors": []}

    # use_document_model=False keeps the raw server JSON.  Changed on 2026-10-23 after the first run: the
    # mp-api 0.46.0 document models failed to validate the database 2026.09.28 documents.
    with MPRester(key, use_document_model=False) as mpr:
        try:
            meta["database_version"] = mpr.get_database_version()
        except Exception:  # noqa: BLE001 - record and continue
            meta["errors"].append({"step": "database_version", "error": traceback.format_exc()})

        summary = []
        try:
            summary = [dump(d) for d in mpr.materials.summary.search(material_ids=ids)]
        except Exception:  # noqa: BLE001
            meta["errors"].append({"step": "table4_summary", "error": traceback.format_exc()})
        write_json(out / "table4_summary.json", summary)

        try:
            thermo = [dump(d) for d in mpr.materials.thermo.search(material_ids=ids)]
        except Exception:  # noqa: BLE001
            thermo = []
            meta["errors"].append({"step": "table4_thermo", "error": traceback.format_exc()})
        write_json(out / "table4_thermo.json", thermo)

        robo = {}
        for mid in ids:
            try:
                # search(keywords=...) searches the description text; search_docs queries by material id
                robo[mid] = [dump(d) for d in mpr.materials.robocrys.search_docs(material_ids=[mid])]
            except Exception:  # noqa: BLE001
                robo[mid] = None
                meta["errors"].append({"step": f"robocrys {mid}", "error": traceback.format_exc()})
        write_json(out / "table4_robocrys.json", robo)

        neighbourhood = []
        try:
            neighbourhood = [dump(d) for d in mpr.materials.summary.search(chemsys=chemsys)]
        except Exception:  # noqa: BLE001
            meta["errors"].append({"step": "neighbourhood_summary", "error": traceback.format_exc()})
        write_json(out / "neighbourhood_summary.json", neighbourhood)

    with (out / "neighbourhood_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["material_id", "formula_pretty", "chemsys", "space_group", "crystal_system",
                    "energy_above_hull", "is_stable", "theoretical", "deprecated"])
        for d in neighbourhood:
            sym = d.get("symmetry") or {}
            w.writerow([d.get("material_id"), d.get("formula_pretty"), d.get("chemsys"), sym.get("symbol"),
                        sym.get("crystal_system"), d.get("energy_above_hull"), d.get("is_stable"),
                        d.get("theoretical"), d.get("deprecated")])

    by_id = {d.get("material_id"): d for d in summary}
    with (out / "table4_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["material_id", "manuscript_formula", "returned_formula", "manuscript_space_group",
                    "returned_space_group", "manuscript_e_above_hull", "returned_e_above_hull", "manuscript_label",
                    "robocrys_description_found"])
        for row in TABLE4:
            d = by_id.get(row["material_id"], {})
            sym = d.get("symmetry") or {}
            w.writerow([row["material_id"], row["formula"], d.get("formula_pretty"), row["space_group"],
                        sym.get("symbol"), row["e_above_hull"], d.get("energy_above_hull"), row["label"],
                        bool(robo.get(row["material_id"]))])

    meta["query_finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    meta["n_table4_docs"] = len(summary)
    meta["n_neighbourhood_docs"] = len(neighbourhood)
    write_json(out / "meta.json", meta)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
