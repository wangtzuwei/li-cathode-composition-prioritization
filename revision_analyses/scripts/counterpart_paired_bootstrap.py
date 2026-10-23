"""Paired bootstrap differences of the counterpart compositions (analysis 05b; see revision_analyses/README.md).

For each of the 7 Table 3 candidates and each of its counterparts (a) no F, (b) no Cr,
(c) no F and no Cr, and Li2MnO3, every one of the 100 saved bootstrap models predicts
both compositions at 298.15 K, 0.5 C, cycle 1 (the Table 3 context) and the difference
candidate - counterpart is taken within the same model.  Nothing is trained.
Outputs: outputs/05b_*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pymatgen.core import Composition

from composition_counterparts import TABLE3, counterparts, load_models
from literature_pairs import features
from candidate_checks import OUT, ssb, write_json


def main() -> None:
    evaluator = ssb.CandidateEvaluator()
    models, feature_columns = load_models()
    if feature_columns != list(evaluator.feature_columns):
        raise RuntimeError("Bootstrap model feature order differs from the evaluator")
    rows, raw = [], {}
    for formula, _, _ in TABLE3:
        others = counterparts(formula) + [("Li2MnO3", "Li2MnO3")]
        formulas = [formula] + [Composition(f).formula.replace(" ", "") for _, f in others]
        X = features(evaluator, formulas, 0.5, 1.0)
        boot = np.column_stack([np.asarray(m.predict(X, thread_count=1), dtype=float) for m in models])
        for j, (label, other) in enumerate(others, start=1):
            diff = boot[0] - boot[j]
            raw[f"{formula} - {label}"] = diff.tolist()
            rows.append({
                "candidate": formula, "comparison": label, "counterpart": formulas[j],
                "median": float(np.median(diff)), "P05": float(np.percentile(diff, 5)),
                "P95": float(np.percentile(diff, 95)), "mean": float(diff.mean()),
                "n_positive": int((diff > 0).sum()), "n_models": int(len(diff)),
                "min": float(diff.min()), "max": float(diff.max()),
            })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "05b_counterpart_paired_bootstrap.csv", index=False, encoding="utf-8-sig")
    write_json("05b_counterpart_paired_bootstrap_raw.json", raw)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(table.round(2).to_string())


if __name__ == "__main__":
    main()
