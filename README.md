# Composition-Level Prioritization of Lithium-Ion Cathode Candidates: code and data

Analysis code, processed data, the fitted CatBoost surrogate and all numerical outputs behind the study *Composition-Level Prioritization of Lithium-Ion Cathode Candidates* (T.-W. Wang, G. Millner, Z. Wang, W. Zhou, E. Y.-S. Chen; manuscript submitted).

The workflow prioritizes cathode compositions for follow-up study. It does not assign polymorphs, establish thermodynamic stability or synthesizability, or validate electrochemical performance.

## Contents

| Path | Content |
|---|---|
| `paper_release_v1/inputs/final.csv` | 907 literature-derived cathode capacity records after manual verification, built from the Huang & Cole battery database (CC BY 4.0; see `LICENSE-DATA`). The 11 structure-property columns are left empty in this release (see `CHANGES.md`). |
| `paper_release_v1/inputs/legacy_features_for_audit.csv` | Legacy element-count features, used only for the composition-identity audit |
| `paper_release_v1/data/` | Canonical composition identities (171 compositions), cleaned records, cross-validation folds, composition-only and composition + context feature matrices (208 features) |
| `paper_release_v1/config/` | Locked package versions, model parameters, feature manifest, search and screening settings |
| `paper_release_v1/models/` | Fitted CatBoost full model (`.cbm` and `.joblib`) |
| `paper_release_v1/results/` | Out-of-fold predictions, model comparisons, leave-one-family-out validation, positive-control calibration, screening, recurrence and uncertainty tables |
| `paper_release_v1/results/search_candidates/` | The matched 20-seed search: NSGA-II and random candidate pools, generation histories, TopK tables for the four arms, per-seed checkpoints |
| `paper_release_v1/figures/` | Figures 1–5 (PDF and PNG), captions, and the source tables in `figures/source_data/` |
| `paper_release_v1/src/` | Analysis scripts |
| `run_analysis.py` | Runs the analysis chain |
| `revision_analyses/` | Scripts and output tables of the additional analyses carried out after submission, with the literature tables they read. Each script and its analysis are listed in `revision_analyses/README.md`. |
| `MANIFEST_SHA256.csv` | Size and SHA-256 of every file |
| `CHANGES.md` | What differs from the authors' internal analysis release |
| `THIRD_PARTY_LICENSES.md` | Source data and software licences |

## How to run

### Environment

Python 3.12.13 with the versions in `paper_release_v1/config/requirements-lock.txt`:

| Package | Version | Package | Version |
|---|---|---|---|
| catboost | 1.2.8 | matminer | 0.9.3 |
| scikit-learn | 1.9.0 | matplotlib | 3.11.1 |
| pandas | 2.3.3 | joblib | 1.5.3 |
| numpy | 2.4.6 | openpyxl | 3.1.5 |
| scipy | 1.18.0 | pillow | 12.2.0 |
| pymatgen | 2025.6.14 | python-docx | 1.2.0 |
| seaborn | 0.13.2 | | |

```bash
conda create -n paper1 python=3.12.13
conda activate paper1
pip install -r paper_release_v1/config/requirements-lock.txt
```

### Run

From the repository root:

```bash
python run_analysis.py --skip-search
```

| Option | Effect |
|---|---|
| (none) | Rebuilds data and models, reruns the 20-seed search (by far the longest step), then the downstream analyses and figures |
| `--skip-search` | Reuses the saved search outputs in `results/search_candidates/` |
| `--skip-models --skip-search` | Also reuses the saved data and model-validation results |

Steps, in order:

1. `rebuild_data_and_models.py`
2. `postprocess_model_validation.py`
3. `search_screen_benchmark.py`
4. `extended_validation.py`
5. `postprocess_final.py`
6. `make_figures.py`

The scripts write into `paper_release_v1/` and overwrite the saved outputs there. Work on a copy if you want to compare against the saved files.

## Reproducibility

The released outputs were produced on Linux (Python 3.12.13). The authors repeated the analysis on Windows 11 with the same locked package versions:

| Check | Result |
|---|---|
| Screening, positive-control calibration, recurrence and the context-robustness table, re-derived from the saved model and the saved 40 candidate pools | All 67 reported values reproduced. The 80 TopK tables were identical in membership and order. The seven recurrent candidates, the 9,815-composition union and the mean pairwise Jaccard (0.005) were identical. |
| Retraining the data pipeline, cross-validation and the 100 bootstrap models | Data counts identical. CatBoost out-of-fold predictions and the 100 bootstrap models were bit-identical. 111 of 112 checked values reproduced after rounding; the exception was the Ridge composition-only Spearman (0.449 vs 0.448). |
| Rerunning the search | Not repeated. Independent seeds share little TopK overlap (mean Jaccard 0.005), so a rerun on another platform is not expected to give the same candidate pools. The saved pools are the reference. |

## Licences

| Material | Licence |
|---|---|
| Code (`*.py`, `paper_release_v1/config/`) | MIT (`LICENSE`) |
| Data, models, results, figures, `revision_analyses/outputs/`, `revision_analyses/literature/` | CC BY 4.0 (`LICENSE-DATA`) |

The records were built from the Huang & Cole database (CC BY 4.0). Cite that database together with this repository. `revision_analyses/outputs/mp_table4/` contains Materials Project data (CC BY 4.0). Third-party licences are listed in `THIRD_PARTY_LICENSES.md`.

## Additional analyses

`revision_analyses/` contains 27 scripts. They read `paper_release_v1/` without modifying it and write tables to `revision_analyses/outputs/`. The saved tables are included. Models retrained by these scripts, and the 100 bootstrap models (regenerated by `revision_analyses/scripts/train_bootstrap_models.py`), are written to `revision_analyses/models_local/` and are not distributed. See `revision_analyses/README.md` for the list of analyses, the inputs that are not distributed, and how to run the scripts.

## Citation and contact

See `CITATION.cff`. The article reference will be added when it is published.

Corresponding authors: Gerfried Millner (gerfried.millner@ntu.edu.sg) and Eason Yi-Sheng Chen (eason.chen@ntu.edu.sg).
