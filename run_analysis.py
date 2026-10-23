#!/usr/bin/env python3
"""Run the analysis chain of paper_release_v1.

Calls the analysis scripts in paper_release_v1/src/ in the order of the original release.  The
original release also contained builders for the Supplementary workbook and Word documents; they
are not part of this package (see CHANGES.md).

Usage (from this folder):
    python run_analysis.py                    # everything, including the 20-seed search (hours)
    python run_analysis.py --skip-search      # reuse results/search_candidates
    python run_analysis.py --skip-models --skip-search
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "paper_release_v1" / "src"


def run(script: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONHASHSEED"] = "0"
    env.setdefault("MPLCONFIGDIR", str(ROOT / "paper_release_v1" / "logs" / "matplotlib"))
    command = [sys.executable, str(SRC / script)]
    print(f"\n>>> {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-models", action="store_true", help="reuse data/ and the model-validation results")
    parser.add_argument("--skip-search", action="store_true", help="reuse results/search_candidates")
    args = parser.parse_args()
    if not args.skip_models:
        run("rebuild_data_and_models.py")
        run("postprocess_model_validation.py")
    if not args.skip_search:
        run("search_screen_benchmark.py")
    run("extended_validation.py")
    run("postprocess_final.py")
    run("make_figures.py")
    print("\nAnalysis steps completed.", flush=True)


if __name__ == "__main__":
    main()
