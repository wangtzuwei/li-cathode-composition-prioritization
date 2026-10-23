# Additional analyses

Scripts for the additional analyses carried out after submission, and the tables they produced. They read the analysis release in `../paper_release_v1/` and never modify it. No search is rerun; some analyses retrain CatBoost models with the release settings, and those models are written to `models_local/`, which is not distributed.

Analysis numbers are internal labels; they also appear as the prefixes of the output files.

| Folder | Content |
|---|---|
| `scripts/` | Analysis scripts |
| `outputs/` | Tables written by the scripts. `outputs/random_forest_reranking/` holds the random-forest re-ranking (`rf_pool_predictions.csv.gz` is gzip-compressed); `outputs/mp_table4/` holds the Materials Project query; `outputs/figure_3a/` holds the redrawn Figure 3A and its source table. |
| `literature/` | Literature tables read by the scripts (values extracted from published articles, with page references) |
| `models_local/` | Created when scripts are run; not distributed |

## Scripts

| No. | Script | Analysis | Outputs |
|---|---|---|---|
| — | `train_bootstrap_models.py` | Regenerates the 100 composition-bootstrap CatBoost models of the release (same generator, seeds and parameters) and saves them to `models_local/bootstrap_models/`. Run this first: the scripts marked † need these models. | `models_local/bootstrap_models/` |
| 03 | `screen_logic_examples.py` | G1–G4 evaluated step by step with the release's own screening functions for the seven candidates, Li2MnO3, and the Cr-only and F-only Li2MnO3 variants (prediction only) | `03_screen_logic_examples.*` |
| 04 | `strict_redox_budget.py` | Strict cation redox budget (Mn to 4+, Ni to 4+, Co to 4+, Fe to 3+, Cr to 6+, ...) for the candidates and positive controls; recurrence under that budget | `04_strict_redox_*` |
| 04 | `screen_without_g4.py` | Screening and recurrence with the G4 capacity gate removed | `04_no_g4_*` |
| 05 | `composition_counterparts.py` † | Counterpart compositions of the seven candidates (without F; without Cr; without both) scored with the 100 bootstrap models; check of the candidate intervals | `05_counterpart_controls.csv`, `05_table3_interval_check.csv` |
| 05 | `random_forest_reranking.py` | Re-ranks the 40 saved candidate pools with a random-forest full model (release RF settings), re-selects TopK, re-screens, recomputes recurrence | `random_forest_reranking/` |
| 05b | `counterpart_paired_bootstrap.py` † | Paired bootstrap differences, candidate minus each counterpart | `05b_*` |
| 06 | `candidate_checks.py` | A: anion-redox share of the predicted capacity. B: influence of the 12 Cr- or F-containing training compositions (retrains). C: cation/anion ratio and framework site limits. D: Li2MnO3-template-only search, NSGA-II vs random. E: per-template and per-class G3 thresholds. F: family-holdout leakage check (retrains). G: motif-level recurrence. | `06A_*` … `06G_*` |
| 06b | `followup_checks.py` | G3 of synthesized Cr/F-substituted materials; site ratios of the controls; leave-one-out influence of individual Cr/F training compositions; family-holdout dose response (retrains) | `06b1_*` … `06b4_*` |
| 06c | `family_leakage_distance.py` | Family holdout after removing untemplated compositions within distance thresholds (retrains) | `06c_*` |
| 08 | `literature_pairs.py` † | Doped-minus-undoped capacity changes in published Cr/F studies vs model predictions | `08_*` |
| 08b | `literature_tables_v2.py` | Builds the v2 literature tables (three further articles) | `literature/*_v2_20261024.csv` |
| 08b | `literature_pairs_v2.py` †, ‡ | Analysis 08 repeated on the v2 table, with the source-paper overlap check | `08b_*` |
| 09 | `training_source_audit.py` ‡ | Traces the 907 training records to their source articles; capacity definitions | `09_*`, `literature/records_to_check.csv` |
| 11 | `literature_processing.py` § | Literature tables: versioned copies, download checklist, Crossref DOI check, candidate BibTeX, external G3 comparison | `11_*`, `literature/candidate_refs.bib` |
| 12 | `li2mno3_template_effects.py` † | Li2MnO3-template training records; model F and Cr effects in four host frameworks; nearest neighbours | `12*` |
| 14 | `test_condition_extraction.py` ¶ | Test conditions of the traced records from the source articles; retraining without high/low upper-voltage records | `14_*`, `literature/test_conditions.csv`, `models_local/upper_voltage_sensitivity_models/` |
| 15, 16 | `site_limits_and_expanded_g3.py` | Site-occupancy limits from the controls; layered G3 threshold with synthesized materials added | `15_*`, `16_*` |
| 17 | `li2_member_analysis.py` † | The Li = 2 member of the candidate motif; motif-level check under the analysis-14 models | `17_*` |
| 18 | `g3_decomposition.py` | Element-wise decomposition of G3; G3 of Li2Mn(1−x)CrxO3 | `18_*` |
| 19 | `mp_table4_query.py` § | Materials Project query for the four entries of Table 4 and the Li–Mn–Cr–O–F neighbourhood (needs `MP_API_KEY`) | `mp_table4/<timestamp>/` |
| 19 | `mp_table4_compare.py` | Compares a saved query with the Table 4 values | `mp_table4/table4_comparison.csv`, `neighbourhood_same_formula.csv` |
| 20 | `local_reliability.py` | Composition-holdout errors by subset; 5-nearest-neighbour distance of the candidates | `20a_*`, `20b_*` |
| 21 | `g3_valence_radii.py` | G3 with charge-balanced, valence-dependent ionic radii; recalibration and re-screening | `21_*` |
| 22 | `context_curves.py` | Predictions vs C-rate and cycle number for two candidates and their counterparts | `22_*` |
| 23 | `model_seed_robustness.py` | Five CatBoost full models with different `random_seed`; re-ranking, screening and recurrence (needs `models_local/pool_features_composition.npz` from `random_forest_reranking.py`) | `23_*` |
| 24 | `experimental_context.py` | Measured vs predicted capacities of synthesized Cr- or F-substituted Li2MnO3 | `24_*` |
| 25 | `precursor_masses.py` | Precursor amounts for 5 g of Li2Mn0.7Cr0.3O2.865F0.135 | `25_*` |
| 27 | `family_holdout_conservative.py` | Family holdout after removing, whenever a family is held out, the untemplated training compositions that clearly belong to it by stoichiometry and chemistry (LiMn1.95Si0.05O4, spinel; Li0.5MnPO4, olivine); a variant also treats six LixMnO2 compositions as layered; the unmodified run is repeated as a check (retrains) | `27_*` |
| 27 | `figure3a_family_holdout.py` | Figure 3A redrawn with the analysis-27 family-holdout value (filled marker) and the release value (open marker), in the style of `make_figures.py` | `figure_3a/` |

Inputs that are not distributed:

- † needs the bootstrap models from `train_bootstrap_models.py`.
- ‡ needs the 2025 data clean-up tables (BatteryDataExtractor records with DOIs); set `BDE_PIPELINE_DIR`. `literature_pairs_v2.py` needs them only for the source-paper overlap check.
- § needs network access (Crossref API; Materials Project API).
- ¶ step 1 needs the source-article PDFs (`PAPER_PDF_ROOT`) and `pdftotext`; step 2 runs from `literature/test_conditions.csv`.

Not included: the check of the manuscript's citation order, which reads the manuscript source.

## Running

Python environment as in the top-level README. From `revision_analyses/scripts/`:

```bash
python train_bootstrap_models.py
python context_curves.py
```

Each script writes into `../outputs/` and overwrites the saved table of the same name. Some scripts read tables written by others (for example, `candidate_checks.py` reads `05_counterpart_controls.csv` and the `04_strict_redox_*` tables), and the saved copies of those tables are included.

The authors re-ran `precursor_masses.py`, `local_reliability.py`, `g3_valence_radii.py`, `context_curves.py`, `screen_logic_examples.py`, `family_holdout_conservative.py` and `figure3a_family_holdout.py` from this folder layout; their outputs were byte-identical to the saved files. Run `figure3a_family_holdout.py` after `family_holdout_conservative.py`.

## Sources

- The literature tables were compiled from published articles. Each value carries its page, table or figure reference. The articles are not redistributed.
- `outputs/mp_table4/` contains records returned by the Materials Project API (database version 2026.09.28, queried 2026-10-22 UTC). Materials Project data are licensed under CC BY 4.0; cite Jain et al., *APL Materials* 1, 011002 (2013), doi:10.1063/1.4812323.
- `literature/candidate_refs.bib` is bibliographic metadata returned by the Crossref API.
