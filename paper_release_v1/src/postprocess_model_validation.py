#!/usr/bin/env python3
"""Create model-only paired inference before the search stage."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from postprocess_release import (
    export_baseline_parameters,
    make_alias_and_integrity_tables,
    make_evaluation_protocol_gap,
    make_inference_family,
    paired_cluster_bootstrap,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "paper_release_v1" / "results"


def main() -> None:
    make_alias_and_integrity_tables()
    export_baseline_parameters()
    oof = pd.read_csv(RESULTS / "oof_canonical_composition_heldout_all_models.csv")
    pairwise = paired_cluster_bootstrap(oof)
    pairwise.to_csv(RESULTS / "paired_model_cluster_bootstrap.csv", index=False)
    gap = make_evaluation_protocol_gap()
    inference = make_inference_family(pairwise, gap)
    path = RESULTS / "results_for_manuscript.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    result["paired_model_comparisons"] = pairwise.to_dict(orient="records")
    result["evaluation_protocol_gap"] = gap
    result["primary_inference_family"] = inference
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Model validation postprocessing complete")


if __name__ == "__main__":
    main()
