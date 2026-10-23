#!/usr/bin/env python3
"""Analysis 19: compare manuscript Table 4 with a saved Materials Project query (read-only).

    python compare_table4_19.py outputs/20261022T235855Z

Database 2026.09.28 returns new-format material ids (e.g. mp-aaaabcci), so query_table4_mp.py's own
table4_comparison.csv has empty returned columns.  The old -> new id map is taken from
table4_robocrys.json, which was queried one old id at a time.  Writes table4_comparison_19.csv and
neighbourhood_same_formula_19.csv into the same output folder; nothing else is modified.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from query_table4_mp import TABLE4

# Table 4 writes Li2Cr2MnO6; MP's formula_pretty writes the same composition as Li2MnCr2O6.
SAME_FORMULA = {"Li2Cr2MnO6": "Li2MnCr2O6"}


def main(folder: Path) -> None:
    load = lambda name: json.loads((folder / name).read_text(encoding="utf-8"))  # noqa: E731
    summary = {d["material_id"]: d for d in load("table4_summary.json")}
    thermo = load("table4_thermo.json")
    robo = load("table4_robocrys.json")
    neighbourhood = load("neighbourhood_summary.json")

    rows = []
    for row in TABLE4:
        docs = robo.get(row["material_id"]) or []
        new_id = docs[0]["material_id"] if docs else None
        d = summary.get(new_id, {})
        mineral = (docs[0].get("condensed_structure") or {}).get("mineral") or {} if docs else {}
        th = {t["thermo_type"]: t["energy_above_hull"] for t in thermo if t["material_id"] == new_id}
        rows.append({
            "manuscript_material_id": row["material_id"], "returned_material_id": new_id,
            "manuscript_formula": row["formula"], "returned_formula": d.get("formula_pretty"),
            "manuscript_space_group": row["space_group"], "returned_space_group": (d.get("symmetry") or {}).get("symbol"),
            "manuscript_label": row["label"], "returned_robocrys_mineral_type": mineral.get("type"),
            "manuscript_e_above_hull": row["e_above_hull"], "returned_e_above_hull": d.get("energy_above_hull"),
            "returned_e_above_hull_GGA_GGA+U": th.get("GGA_GGA+U"),
            "returned_e_above_hull_GGA_GGA+U_R2SCAN": th.get("GGA_GGA+U_R2SCAN"),
            "returned_e_above_hull_r2SCAN": th.get("r2SCAN"),
            "returned_is_stable": d.get("is_stable"), "returned_theoretical": d.get("theoretical"),
            "returned_deprecated": d.get("deprecated"), "returned_last_updated": d.get("last_updated"),
            "returned_database_version": (d.get("builder_meta") or {}).get("database_version"),
            "robocrys_description_start": docs[0].get("description", "")[:120] if docs else None,
        })
    with (folder / "table4_comparison_19.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # every entry in the neighbourhood query with the same formula as a Table 4 row
    with (folder / "neighbourhood_same_formula_19.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["table4_formula", "material_id", "space_group", "energy_above_hull", "is_stable",
                    "theoretical", "is_table4_entry"])
        table4_new = {r["returned_material_id"] for r in rows}
        for row in TABLE4:
            pretty = SAME_FORMULA.get(row["formula"], row["formula"])
            same = sorted((d for d in neighbourhood if d.get("formula_pretty") == pretty),
                          key=lambda d: d.get("energy_above_hull") if d.get("energy_above_hull") is not None else 9e9)
            for d in same:
                w.writerow([row["formula"], d["material_id"], (d.get("symmetry") or {}).get("symbol"),
                            d.get("energy_above_hull"), d.get("is_stable"), d.get("theoretical"),
                            d["material_id"] in table4_new])
    print(f"Wrote {folder / 'table4_comparison_19.csv'} and neighbourhood_same_formula_19.csv")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
