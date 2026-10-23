# Differences from the authors' internal analysis release

`paper_release_v1/` is a subset of the authors' internal analysis release (`paper_release_v1`, 26 July 2026). Every file kept from that release is byte-identical, except the three listed under "Edited". `revision_analyses/` is new.

## Not included

| Item | Reason |
|---|---|
| `src/build_documents.py`, `src/build_final_documents_v2.py`, `src/build_delivery_package.py`, `src/build_final_package.py`, `src/validate_final_rebuild.py`, `src/validate_release.py` | Build or check the manuscript and internal documents from a manuscript draft that is not distributed |
| `src/update_supplementary_workbook.py`, `src/prepare_workbook_spec.py`, `src/build_final_workbook.py`, `src/build_final_workbook.mjs`, `src/update_release_readme.py`, `run_all.py`, `results/workbook_spec.json`, `results/workbook_intermediate/` | Build the Supplementary workbook and release notes. The workbook step needs a JavaScript runtime that existed only in the original computing environment. The workbook sheets are copies of tables already in `results/`. |
| `inputs/LEGACY_SOURCE_NOTE.md` and the legacy notebooks it describes | Earlier exploratory notebooks, not part of the analysis chain |
| `figures/source_assets/Figure2–5_author_final_reference.png` | Reference images; not read by any script |
| Manuscript drafts | Not distributed |

`run_analysis.py` runs the same analysis scripts, in the same order, as the original `run_all.py`, and stops before the document and workbook builders.

## Edited

- `paper_release_v1/src/make_figures.py`:
  - one code comment reworded;
  - the fallback Matplotlib configuration directory changed from `/tmp/mpl-author-final` to a folder in the system temporary directory (`tempfile.gettempdir()`), so that it works on any operating system.

  The figure output is not affected.
- `paper_release_v1/inputs/final.csv` and `paper_release_v1/data/clean_rows.csv`: the 11 structure-property columns are left empty in this release. The headers are kept and every other byte is unchanged. The columns are `crystal_system`, `space_group_number`, `sites`, `E_above_hull(eV/atom)`, `formationE(eV/atom)`, `volume(A^3)`, `density(g/cm^3)`, `band_gap(eV)`, `is_gap_direct`, `magnetic_ordering` and `tot_magnetization`; they were filled for 634 of the 907 records. No analysis reads these columns, so features, models and results are unaffected.

## Added: `revision_analyses/`

Scripts and output tables of the additional analyses carried out after submission, taken from the authors' working folders. Compared with the working copies:

- **Names:** scripts and two output tables were renamed to describe the analysis (for example, `05_counterpart_controls.csv`). File names and comments no longer carry review-item labels.
- **Paths:** paths were rewritten for this repository layout. Inputs that are not distributed are read from environment variables (`BDE_PIPELINE_DIR`, `PAPER_PDF_ROOT`) instead of local folders.
- **Model writes:** scripts that retrain models write them to `revision_analyses/models_local/` (not distributed). `train_bootstrap_models.py` writes its uncertainty tables there instead of into `paper_release_v1/results/`.
- **Not included:**
  - the check of the manuscript's citation order;
  - the raw Crossref responses (`11_crossref_raw.json`, which contain article abstracts);
  - the raw PDF text snippets of analysis 14 (`14_candidates.json`);
  - retrained models and cached feature matrices.
- **Literature tables:** local PDF folder paths in the notes were shortened to `literature/pdfs/`; the PDFs themselves are not distributed.
- **Compressed output:** the random-forest pool predictions are stored gzip-compressed (`rf_pool_predictions.csv.gz`).

## Added on 24 October 2026 (before the first Zenodo publication)

- `revision_analyses/scripts/screen_logic_examples.py` (analysis 03) and its tables `outputs/03_screen_logic_examples.csv` and `.json` (renamed with the analysis prefix; content unchanged).
- `revision_analyses/scripts/family_holdout_conservative.py` (analysis 27) and its tables `outputs/27_*`.
- `revision_analyses/scripts/figure3a_family_holdout.py`, which redraws Figure 3A with the analysis-27 value, and its output in `outputs/figure_3a/` (`Figure_3A_revised.png` and the source table `Figure3A_validation_hierarchy_revised.csv`).

These three scripts were run from this folder layout; all their outputs were byte-identical to the saved files.
