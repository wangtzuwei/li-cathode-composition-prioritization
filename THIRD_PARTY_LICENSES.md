# Third-party data and software

## Data

| Source | Used for | Licence |
|---|---|---|
| Huang, S. & Cole, J. M. *A database of battery materials auto-generated using ChemDataExtractor*. *Scientific Data* 7, 260 (2020), doi:10.1038/s41597-020-00602-2. Dataset: Figshare, doi:10.6084/m9.figshare.11888115.v2 | Starting records of `inputs/final.csv` (names, values, units, specifiers, conditions; later cleaned and manually verified) | CC BY 4.0 (Figshare record licence field; the article is also CC BY 4.0; checked 2026-10-23) |
| Values re-read from the original journal articles during the 2025 manual verification | Corrected capacities, compositions and conditions in `inputs/final.csv` | Numerical facts. The articles themselves are not redistributed. |
| Magpie elemental property tables, as distributed with matminer | Composition descriptors | Part of matminer (modified BSD) |
| Materials Project (Jain et al., *APL Materials* 1, 011002 (2013), doi:10.1063/1.4812323), database version 2026.09.28 | `revision_analyses/outputs/mp_table4/` (query of the Table 4 entries and the Li–Mn–Cr–O–F neighbourhood, 2026-10-22 UTC) | CC BY 4.0 |
| Crossref REST API | Bibliographic metadata in `revision_analyses/literature/candidate_refs.bib` and the DOI check of analysis 11 | Crossref metadata are made available without restriction (CC0) |
| Published journal articles | Values in `revision_analyses/literature/` (compositions, capacities, test and synthesis conditions, with page references) | Numerical facts with short quoted fragments as page references. The articles themselves are not redistributed. |

## Software

Python packages, with versions locked in `paper_release_v1/config/requirements-lock.txt`. The licence is the one declared in each installed package's metadata (checked 2026-10-24).

| Package | Version | Licence |
|---|---|---|
| catboost | 1.2.8 | Apache-2.0 |
| scikit-learn | 1.9.0 | BSD-3-Clause |
| pandas | 2.3.3 | BSD-3-Clause |
| numpy | 2.4.6 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| scipy | 1.18.0 | BSD (classifier "BSD License") |
| pymatgen | 2025.6.14 | MIT |
| matminer | 0.9.3 | modified BSD |
| matplotlib | 3.11.1 | Matplotlib licence (PSF-based) |
| joblib | 1.5.3 | BSD-3-Clause |
| openpyxl | 3.1.5 | MIT |
| pillow | 12.2.0 | MIT-CMU |
| python-docx | 1.2.0 | MIT |
| seaborn | 0.13.2 | BSD (classifier "BSD License") |

These packages are installed by the user and are not redistributed in this repository.
