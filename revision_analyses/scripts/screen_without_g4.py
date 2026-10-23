"""Scenario: drop G4 entirely (G1, G2 and G3 at P95 unchanged).

Same procedure as strict_redox_budget.recurrence(): reads the 20 saved reference
TopK tables (nsga2__floor_on), re-applies the screen with G4 removed, recomputes
recurrence with the original rule, and counts positive-control retention from the
saved known_control_screened.csv.  Saved artifacts only; nothing is trained or
searched and no existing program is changed.
Outputs: outputs/no_g4_*.
"""
from __future__ import annotations

import json

import pandas as pd

from strict_redox_budget import OUT, SAVED, SEEDS


def main() -> None:
    known_summary = json.loads((SAVED / "results" / "known_control_summary.json").read_text(encoding="utf-8"))
    p95 = float(known_summary["positive_control_radius_p95_after_g1_g2"])

    frames = []
    for seed in SEEDS:
        frame = pd.read_csv(SAVED / "results" / "search_candidates" / f"nsga2__floor_on_seed{seed}_topk_screened.csv")
        frame["seed"] = seed
        radius = pd.to_numeric(frame["radius_mismatch"], errors="coerce")
        g12 = frame["g1_pass"].astype(bool) & frame["g2_pass"].astype(bool)
        frame["p95_final_original"] = g12 & frame["g4_pass"].astype(bool) & radius.le(p95)
        frame["p95_final_no_g4"] = g12 & radius.le(p95)
        frames.append(frame)
    rows = pd.concat(frames, ignore_index=True)

    records = []
    for key, part in rows.groupby("canonical_key", sort=True):
        seeds = sorted(part["seed"].unique())
        passing = sorted(part.loc[part["p95_final_no_g4"], "seed"].unique())
        records.append({
            "canonical_key": key, "Formula": part["Formula"].iloc[0], "proto_name": part["proto_name"].iloc[0],
            "topk_seed_count": len(seeds), "04_no_g4_p95_final_seed_count": len(passing),
            "y_pred_mean": float(part["y_pred"].mean()),
            "consensus_no_g4": (len(seeds) >= 10 and len(passing) == len(seeds)
                                and bool(part["g1_pass"].astype(bool).all())
                                and bool(part["g2_pass"].astype(bool).all())),
        })
    union = pd.DataFrame(records)
    union.to_csv(OUT / "04_no_g4_recurrence_union.csv", index=False)
    consensus = union[union["consensus_no_g4"]].sort_values("topk_seed_count", ascending=False)
    original = pd.read_csv(SAVED / "results" / "consensus_candidates.csv")
    per_seed = rows.groupby("seed")[["p95_final_original", "p95_final_no_g4"]].sum().astype(int)

    known = pd.read_csv(SAVED / "results" / "known_control_screened.csv")
    known_radius = pd.to_numeric(known["radius_mismatch"], errors="coerce")
    known_g12 = known["g1_pass"].astype(bool) & known["g2_pass"].astype(bool)
    controls_no_g4 = known_g12 & known_radius.le(p95)
    controls_original = controls_no_g4 & known["g4_pass"].astype(bool)
    by_proto = pd.DataFrame({
        "proto_name": known["proto_name"], "retained_original": controls_original, "retained_no_g4": controls_no_g4,
    }).groupby("proto_name").agg(n=("retained_original", "size"), retained_original=("retained_original", "sum"),
                                 retained_no_g4=("retained_no_g4", "sum")).reset_index()

    summary = {
        "p95": p95,
        "per_seed_mean_original": float(per_seed["p95_final_original"].mean()),
        "per_seed_mean_no_g4": float(per_seed["p95_final_no_g4"].mean()),
        "per_seed_p95_survivors": per_seed.reset_index().to_dict(orient="records"),
        "union": int(len(union)),
        "candidates_with_10plus_seeds": int((union["topk_seed_count"] >= 10).sum()),
        "consensus_no_g4": consensus[["Formula", "proto_name", "topk_seed_count", "y_pred_mean"]].to_dict(orient="records"),
        "original_retained": sorted(set(original["Formula"]) & set(consensus["Formula"])),
        "original_dropped": sorted(set(original["Formula"]) - set(consensus["Formula"])),
        "new_recurrent": sorted(set(consensus["Formula"]) - set(original["Formula"])),
        "controls_total": int(len(known)),
        "controls_retained_original": int(controls_original.sum()),
        "controls_retained_no_g4": int(controls_no_g4.sum()),
        "controls_by_prototype": by_proto.to_dict(orient="records"),
        "candidates_failing_only_g4_per_seed_mean": float(
            (rows["p95_final_no_g4"] & ~rows["g4_pass"].astype(bool)).groupby(rows["seed"]).sum().mean()),
    }
    (OUT / "04_no_g4_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_seed_p95_survivors"}, indent=1, default=float))
    print(per_seed.to_string())


if __name__ == "__main__":
    main()
