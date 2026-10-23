"""25: precursor amounts for 5 g of Li2Mn0.7Cr0.3O2.865F0.135 (analysis 25; see revision_analyses/README.md).

Molar masses from pymatgen (IUPAC standard atomic weights).  Per formula unit (f.u.) of the target:
  Solid state (Li2CO3, MnCO3, Cr2O3, LiF):
    F only from LiF: LiF = 0.135 x (1 + e), e = 0 / 0.15 / 0.30 (F loss 13-33 % in Giovine 2024).
    Li excess 5 %: two bookkeepings are listed.
      A  total Li fixed at 2 x 1.05 = 2.10; the extra LiF Li counts toward it (Li2CO3 = (2.10 - LiF) / 2).
      B  Li2CO3 carries 5 % excess on the Li not supplied by the nominal LiF, (2 - 0.135) x 1.05 / 2;
         the extra LiF is added on top.
    Mn = 0.7 MnCO3, Cr = 0.3 / 2 Cr2O3.  O is not balanced: carbonate decomposition and the furnace
    atmosphere set the O content.
  Ball milling (Li2O, MnO2, Cr2O3, LiF), no F loss, no Li excess (stoichiometric):
    LiF = 0.135, Li2O = (2 - 0.135) / 2, MnO2 = 0.7, Cr2O3 = 0.15.  The O delivered by these precursors is
    reported next to the target O.
Outputs: outputs/25_precursors.csv, outputs/25_summary.json.
"""
from __future__ import annotations

import pandas as pd
from pymatgen.core import Composition

from candidate_checks import OUT, write_json

TARGET = "Li2Mn0.7Cr0.3O2.865F0.135"
PRODUCT_G = 5.0
LI, F, MN, CR, O = 2.0, 0.135, 0.7, 0.3, 2.865


def mm(formula: str) -> float:
    return float(Composition(formula).weight)


def rows_for(route: str, recipe: dict[str, float], n_fu: float) -> list[dict[str, object]]:
    out = []
    for precursor, per_fu in recipe.items():
        out.append({"route": route, "precursor": precursor, "mol_per_fu": per_fu, "molar_mass_g_mol": mm(precursor),
                    "mol_for_5g": per_fu * n_fu, "mass_g": per_fu * n_fu * mm(precursor)})
    return out


def delivered(recipe: dict[str, float]) -> dict[str, float]:
    total: dict[str, float] = {}
    for precursor, per_fu in recipe.items():
        for el, amt in Composition(precursor).get_el_amt_dict().items():
            total[el] = total.get(el, 0.0) + amt * per_fu
    return total


def main() -> None:
    m_target = mm(TARGET)
    n_fu = PRODUCT_G / m_target
    rows, checks = [], {}
    for excess in [0.0, 0.15, 0.30]:
        lif = F * (1 + excess)
        for book, li2co3 in [("A", (LI * 1.05 - lif) / 2), ("B", (LI - F) * 1.05 / 2)]:
            route = f"固相 LiF +{excess:.0%}／Li 記帳 {book}"
            recipe = {"Li2CO3": li2co3, "MnCO3": MN, "Cr2O3": CR / 2, "LiF": lif}
            rows += rows_for(route, recipe, n_fu)
            checks[route] = {k: round(v, 6) for k, v in delivered(recipe).items() if k in ("Li", "Mn", "Cr", "F")}
    milling = {"Li2O": (LI - F) / 2, "MnO2": MN, "Cr2O3": CR / 2, "LiF": F}
    rows += rows_for("球磨", milling, n_fu)
    d = delivered(milling)
    checks["球磨"] = {k: round(v, 6) for k, v in d.items()}
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "25_precursors.csv", index=False, encoding="utf-8-sig")
    # charge check of the milling product: Li+ + Mn4+ + Cr3+ against O2- and F-
    o_from_charge = (LI + 4 * MN + 3 * CR - F) / 2
    write_json("25_summary.json", {
        "target": TARGET, "target_molar_mass": m_target, "product_g": PRODUCT_G, "mol_fu_for_5g": n_fu,
        "delivered_per_fu": checks, "milling_O_delivered_per_fu": d["O"], "target_O_per_fu": O,
        "O_for_Li+_Mn4+_Cr3+_and_F0.135": o_from_charge,
        "average_TM_valence_needed_for_O2.865": (2 * O + F - LI) / (MN + CR),
        "molar_masses": {p: mm(p) for p in ["Li2CO3", "MnCO3", "Cr2O3", "LiF", "Li2O", "MnO2"]}})
    print(f"M = {m_target:.4f} g/mol, n = {n_fu:.6f} mol")
    print(table.round(5).to_string())
    print(checks)


if __name__ == "__main__":
    main()
