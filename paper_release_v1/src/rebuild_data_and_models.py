#!/usr/bin/env python3
"""Canonical data rebuild and leakage-resistant model evaluation for Paper 1.

This script deliberately ignores the legacy ``Extracted_name`` field when
constructing chemical identity and composition descriptors.  The raw ``formula``
column is parsed with pymatgen, converted to a scale/order-invariant fractional
composition key, and used for every group-held-out analysis.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from matminer.featurizers.composition import ElementProperty, Stoichiometry
from pymatgen.core import Composition
from scipy.stats import rankdata, spearmanr
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "project_sources"
RELEASE = ROOT / "paper_release_v1"
INPUT_DIR = RELEASE / "inputs"
DATA_DIR = RELEASE / "data"
CONFIG_DIR = RELEASE / "config"
RESULTS_DIR = RELEASE / "results"
MODELS_DIR = RELEASE / "models"
LOG_DIR = RELEASE / "logs"

for directory in [DATA_DIR, CONFIG_DIR, RESULTS_DIR, MODELS_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

RAW_PATH = INPUT_DIR / "final.csv" if (INPUT_DIR / "final.csv").exists() else SOURCE_DIR / "14-final.csv"
LEGACY_FEATURE_PATH = (
    INPUT_DIR / "legacy_features_for_audit.csv"
    if (INPUT_DIR / "legacy_features_for_audit.csv").exists()
    else SOURCE_DIR / "06-features_componly.csv"
)

SEED = 42
N_SPLITS = 10
BOOTSTRAP_REPS = 5000
RHO_BOOTSTRAP_REPS = 10000
PERMUTATION_REPS = 100000

PERIODIC_ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
]

ANIONS = {"O", "F", "S", "Cl", "Br", "I"}
TM_SITE = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Y", "Zr", "Nb", "Mo", "Tc", "Ru",
    "Rh", "Pd", "Ag", "Cd", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Al", "Mg", "Ga", "Sn", "Ge",
}

CONTEXT_COLUMNS = [
    "temperature_K",
    "log1p_current_rate_C",
    "log1p_cycle_number",
    "log1p_current_rate_x_cycle",
]

CB_PARAMS = {
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 3.0,
    "loss_function": "RMSE",
    "random_seed": SEED,
    "allow_writing_files": False,
    "verbose": False,
    "thread_count": 4,
}


def log(message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]", "_", str(text))
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "feature"


def canonical_key(comp: Composition, decimals: int = 8) -> str:
    fractions = comp.fractional_composition.get_el_amt_dict()
    return "|".join(f"{element}:{fractions[element]:.{decimals}f}" for element in sorted(fractions))


def fractional_dict(comp: Composition) -> dict[str, float]:
    return {str(k): float(v) for k, v in comp.fractional_composition.get_el_amt_dict().items()}


def parse_extracted_name(value: object) -> dict[str, float]:
    """Parse the legacy stringified list/dict only for auditing, never modeling."""
    import ast

    try:
        obj = ast.literal_eval(str(value))
    except Exception:
        return {}
    blocks = obj if isinstance(obj, list) else [obj]
    output: dict[str, float] = {}
    for block in blocks:
        if not isinstance(block, dict):
            continue
        for element, amount in block.items():
            try:
                output[str(element)] = output.get(str(element), 0.0) + float(amount)
            except Exception:
                continue
    return output


def normalize_dict(values: dict[str, float], *, exclude_carbon: bool = False) -> dict[str, float]:
    cleaned = {
        key: float(value)
        for key, value in values.items()
        if float(value) > 0 and not (exclude_carbon and key == "C")
    }
    total = sum(cleaned.values())
    return {key: value / total for key, value in cleaned.items()} if total > 0 else {}


def max_fraction_difference(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    return max((abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys), default=0.0)


@dataclass(frozen=True)
class FeatureArtifacts:
    composition: pd.DataFrame
    context: pd.DataFrame
    manifest: pd.DataFrame


def build_canonical_table(raw: pd.DataFrame, legacy_features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    legacy_elem_cols = [column for column in legacy_features if column.startswith("elemcnt_")]
    for row_id, row in raw.iterrows():
        formula = str(row["formula"])
        parse_error = ""
        try:
            comp = Composition(formula.strip())
        except Exception as exc:  # no current rows should enter this branch
            comp = None
            parse_error = str(exc)

        if comp is None:
            rows.append(
                {
                    "row_id": row_id,
                    "raw_formula": formula,
                    "parse_ok": False,
                    "parse_error": parse_error,
                }
            )
            continue

        raw_fraction = fractional_dict(comp)
        extracted = parse_extracted_name(row.get("Extracted_name"))
        extracted_fraction = normalize_dict(extracted, exclude_carbon=True)

        legacy_counts = {
            column.removeprefix("elemcnt_"): float(legacy_features.loc[row_id, column])
            for column in legacy_elem_cols
            if float(legacy_features.loc[row_id, column]) > 0
        }
        legacy_fraction = normalize_dict(legacy_counts, exclude_carbon=True)

        rows.append(
            {
                "row_id": int(row_id),
                "raw_formula": formula,
                "whitespace_normalized_formula": re.sub(r"\s+", "", formula),
                "canonical_formula": comp.formula.replace(" ", ""),
                "reduced_formula": comp.reduced_formula.replace(" ", ""),
                "canonical_key_6": canonical_key(comp, 6),
                "canonical_key_8": canonical_key(comp, 8),
                "canonical_key_10": canonical_key(comp, 10),
                "parse_ok": True,
                "parse_error": "",
                "raw_extracted_name": row.get("Extracted_name"),
                "carbon_in_extracted_name": float(extracted.get("C", 0.0)) > 0,
                "raw_vs_extracted_max_fraction_diff": max_fraction_difference(raw_fraction, extracted_fraction),
                "raw_vs_extracted_mismatch": max_fraction_difference(raw_fraction, extracted_fraction) > 1e-6,
                "raw_vs_legacy_feature_max_fraction_diff": max_fraction_difference(raw_fraction, legacy_fraction),
                "raw_vs_legacy_feature_mismatch": max_fraction_difference(raw_fraction, legacy_fraction) > 1e-6,
                "legacy_zero_element_vector": len(legacy_counts) == 0,
                "source_value": float(row["Value"]),
                "source_temperature_K": float(row["temperature"]),
                "source_current_rate_C": float(row["current_rate"]),
                "source_cycle_number": float(row["cycle_value"]),
            }
        )

    audit = pd.DataFrame(rows)
    if not audit["parse_ok"].all():
        failed = audit.loc[~audit["parse_ok"], ["row_id", "raw_formula", "parse_error"]]
        failed.to_csv(DATA_DIR / "quarantined_rows.csv", index=False)
        raise RuntimeError(f"Formula parsing failed for {len(failed)} rows; see quarantined_rows.csv")
    pd.DataFrame(columns=["row_id", "raw_formula", "parse_error"]).to_csv(
        DATA_DIR / "quarantined_rows.csv", index=False
    )
    return audit


def build_features(raw: pd.DataFrame, audit: pd.DataFrame) -> FeatureArtifacts:
    stoich = Stoichiometry()
    magpie = ElementProperty.from_preset("magpie")
    stoich_labels = [sanitize(f"Stoich_{label}") for label in stoich.feature_labels()]
    # ``range`` is exactly max-min and therefore redundant when both extrema
    # are present.  Dropping the 22 range columns reproduces the intended
    # 110-column Magpie block without data-dependent feature selection.
    magpie_raw_labels = list(magpie.feature_labels())
    magpie_labels = [
        sanitize(f"Magpie_{label}")
        for label in magpie_raw_labels
        if "_range_" not in sanitize(f"Magpie_{label}")
    ]
    magpie_label_set = set(magpie_labels)

    unique_keys = audit[["canonical_key_8", "raw_formula"]].drop_duplicates("canonical_key_8")
    cache: dict[str, dict[str, float]] = {}
    for counter, row in enumerate(unique_keys.itertuples(index=False), 1):
        comp = Composition(str(row.raw_formula).strip())
        fractions = fractional_dict(comp)
        values: dict[str, float] = {
            f"elemfrac_{element}": float(fractions.get(element, 0.0)) for element in PERIODIC_ELEMENTS
        }
        for label, value in zip(stoich_labels, stoich.featurize(comp)):
            values[label] = float(value)
        for raw_label, value in zip(magpie_raw_labels, magpie.featurize(comp)):
            label = sanitize(f"Magpie_{raw_label}")
            if label in magpie_label_set:
                values[label] = float(value)
        values["anion_fraction_sum"] = float(sum(fractions.get(element, 0.0) for element in ANIONS))
        values["tm_site_fraction_sum"] = float(sum(fractions.get(element, 0.0) for element in TM_SITE))
        cache[str(row.canonical_key_8)] = values
        if counter % 50 == 0:
            log(f"Featurized {counter}/{len(unique_keys)} canonical compositions")

    composition_rows = [cache[key].copy() for key in audit["canonical_key_8"]]
    X_composition = pd.DataFrame(composition_rows, index=raw.index)

    current_rate = pd.to_numeric(raw["current_rate"], errors="coerce").astype(float)
    cycle = pd.to_numeric(raw["cycle_value"], errors="coerce").astype(float)
    context = pd.DataFrame(
        {
            "temperature_K": pd.to_numeric(raw["temperature"], errors="coerce").astype(float),
            "log1p_current_rate_C": np.log1p(current_rate.clip(lower=0)),
            "log1p_cycle_number": np.log1p(cycle.clip(lower=0)),
            "log1p_current_rate_x_cycle": np.log1p((current_rate * cycle).clip(lower=0)),
        },
        index=raw.index,
    )
    X_context = pd.concat([X_composition, context], axis=1)

    records = []
    for column in X_context.columns:
        if column.startswith("elemfrac_"):
            category, definition = "element_fraction", "Atomic fraction from canonical parsed formula"
        elif column.startswith("Stoich_"):
            category, definition = "stoichiometry", "matminer Stoichiometry descriptor"
        elif column.startswith("Magpie_"):
            category, definition = "magpie", "matminer Magpie ElementProperty statistic"
        elif column in CONTEXT_COLUMNS:
            category, definition = "measurement_context", "Recorded electrochemical test context"
        else:
            category, definition = "composition_engineered", "Composition/curation-derived scalar"
        records.append(
            {
                "feature": column,
                "category": category,
                "definition": definition,
                "included_composition_only": column not in CONTEXT_COLUMNS,
                "included_context_model": True,
                "constant_in_full_data": bool(X_context[column].nunique(dropna=False) <= 1),
                "missing_count": int(X_context[column].isna().sum()),
            }
        )
    manifest = pd.DataFrame(records)
    return FeatureArtifacts(X_composition, X_context, manifest)


def catboost_factory() -> CatBoostRegressor:
    return CatBoostRegressor(**CB_PARAMS)


def model_factories() -> dict[str, Callable[[], object]]:
    return {
        "dummy_median": lambda: Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("model", DummyRegressor(strategy="median"))]
        ),
        "ridge": lambda: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest": lambda: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        min_samples_leaf=2,
                        max_features=1.0,
                        random_state=SEED,
                        n_jobs=4,
                    ),
                ),
            ]
        ),
        "catboost": catboost_factory,
    }


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "spearman": float(spearmanr(y_true, y_pred).statistic),
    }


def formula_balanced_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray
) -> dict[str, float]:
    frame = pd.DataFrame({"group": groups, "y_true": y_true, "y_pred": y_pred})
    grouped = frame.groupby("group", sort=False)[["y_true", "y_pred"]].mean()
    return metrics(grouped["y_true"].to_numpy(), grouped["y_pred"].to_numpy())


def bootstrap_metrics_by_group(
    frame: pd.DataFrame, n_reps: int = BOOTSTRAP_REPS, seed: int = SEED
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    groups = frame["canonical_key"].unique()
    by_group = {group: frame.index[frame["canonical_key"] == group].to_numpy() for group in groups}
    values = {name: [] for name in ["r2", "mae", "rmse", "spearman"]}
    for _ in range(n_reps):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([by_group[group] for group in sampled_groups])
        m = metrics(frame.loc[indices, "y_true"].to_numpy(), frame.loc[indices, "y_pred"].to_numpy())
        for name, value in m.items():
            if np.isfinite(value):
                values[name].append(value)
    return {
        name: [float(np.percentile(items, 2.5)), float(np.percentile(items, 97.5))]
        for name, items in values.items()
    }


def fold_consistent_distances(
    X: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, k: int = 5
) -> tuple[np.ndarray, float]:
    X_train = X.iloc[train_idx].apply(pd.to_numeric, errors="coerce")
    X_test = X.iloc[test_idx].apply(pd.to_numeric, errors="coerce")
    imputer = SimpleImputer(strategy="median")
    train_imp = imputer.fit_transform(X_train)
    test_imp = imputer.transform(X_test)

    nonconstant = np.nanstd(train_imp, axis=0) > 1e-12
    train_imp = train_imp[:, nonconstant]
    test_imp = test_imp[:, nonconstant]
    scaler = RobustScaler()
    train_scaled = scaler.fit_transform(train_imp)
    test_scaled = scaler.transform(test_imp)

    n_neighbors_train = min(len(train_scaled), k + 1)
    nn_train = NearestNeighbors(n_neighbors=n_neighbors_train, metric="euclidean", n_jobs=4)
    nn_train.fit(train_scaled)
    train_distances, _ = nn_train.kneighbors(train_scaled)
    train_use = train_distances[:, 1:] if train_distances.shape[1] > 1 else train_distances
    train_mean = train_use.mean(axis=1)
    d95 = float(np.percentile(train_mean, 95))

    nn_test = NearestNeighbors(n_neighbors=min(k, len(train_scaled)), metric="euclidean", n_jobs=4)
    nn_test.fit(train_scaled)
    test_distances, _ = nn_test.kneighbors(test_scaled)
    test_mean = test_distances.mean(axis=1)
    normalized = test_mean / max(d95, 1e-12)
    return normalized, d95


def evaluate_record_cv(X: pd.DataFrame, y: pd.Series, groups: np.ndarray) -> pd.DataFrame:
    bins = pd.qcut(y, q=10, labels=False, duplicates="drop")
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    predictions = np.full(len(y), np.nan)
    fold_ids = np.full(len(y), -1, dtype=int)
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, bins), 1):
        model = catboost_factory()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        predictions[test_idx] = model.predict(X.iloc[test_idx])
        fold_ids[test_idx] = fold
        log(f"Record-CV CatBoost fold {fold}/{N_SPLITS} complete")
    return pd.DataFrame(
        {
            "row_id": np.arange(len(y)),
            "canonical_key": groups,
            "fold_id": fold_ids,
            "y_true": y.to_numpy(float),
            "y_pred": predictions,
        }
    )


def evaluate_logo(
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    model_name: str,
    factory: Callable[[], object],
    with_distances: bool = False,
) -> pd.DataFrame:
    logo = LeaveOneGroupOut()
    prediction = np.full(len(y), np.nan)
    distance = np.full(len(y), np.nan)
    d95_values = np.full(len(y), np.nan)
    fold_id = np.full(len(y), -1, dtype=int)
    n_folds = len(np.unique(groups))
    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), 1):
        model = factory()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        prediction[test_idx] = model.predict(X.iloc[test_idx])
        fold_id[test_idx] = fold
        if with_distances:
            distance[test_idx], d95 = fold_consistent_distances(X, train_idx, test_idx)
            d95_values[test_idx] = d95
        if fold % 20 == 0 or fold == n_folds:
            log(f"LOCO {model_name}: fold {fold}/{n_folds}")
    return pd.DataFrame(
        {
            "row_id": np.arange(len(y)),
            "canonical_key": groups,
            "fold_id": fold_id,
            "model": model_name,
            "y_true": y.to_numpy(float),
            "y_pred": prediction,
            "abs_error": np.abs(y.to_numpy(float) - prediction),
            "normalized_5nn_distance": distance,
            "training_d95": d95_values,
        }
    )


def distance_inference(oof: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    formula = (
        oof.groupby("canonical_key", as_index=False)
        .agg(
            n_records=("row_id", "size"),
            median_abs_error=("abs_error", "median"),
            median_normalized_5nn_distance=("normalized_5nn_distance", "median"),
        )
        .sort_values("canonical_key")
        .reset_index(drop=True)
    )
    x = formula["median_normalized_5nn_distance"].to_numpy(float)
    y = formula["median_abs_error"].to_numpy(float)
    observed = float(spearmanr(x, y).statistic)

    rng = np.random.default_rng(SEED)
    boot = np.empty(RHO_BOOTSTRAP_REPS, dtype=float)
    n = len(formula)
    for idx in range(RHO_BOOTSTRAP_REPS):
        sample = rng.integers(0, n, size=n)
        boot[idx] = spearmanr(x[sample], y[sample]).statistic
    boot = boot[np.isfinite(boot)]

    x_rank = rankdata(x)
    y_rank = rankdata(y)
    xz = (x_rank - x_rank.mean()) / x_rank.std(ddof=0)
    yz = (y_rank - y_rank.mean()) / y_rank.std(ddof=0)
    extreme = 0
    done = 0
    batch = 5000
    while done < PERMUTATION_REPS:
        current = min(batch, PERMUTATION_REPS - done)
        permuted = np.vstack([rng.permutation(yz) for _ in range(current)])
        correlations = (permuted @ xz) / n
        extreme += int((np.abs(correlations) >= abs(observed)).sum())
        done += current
    p_value = (extreme + 1) / (PERMUTATION_REPS + 1)

    result = {
        "n_canonical_compositions": int(n),
        "spearman_rho": observed,
        "rho_bootstrap_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "formula_label_permutation_p_two_sided": float(p_value),
        "bootstrap_repetitions": RHO_BOOTSTRAP_REPS,
        "permutation_repetitions": PERMUTATION_REPS,
    }
    return formula, result


def save_environment() -> None:
    packages = [
        "catboost",
        "joblib",
        "matplotlib",
        "matminer",
        "numpy",
        "openpyxl",
        "pandas",
        "pillow",
        "pymatgen",
        "python-docx",
        "scikit-learn",
        "seaborn",
        "scipy",
    ]
    lines = []
    versions = {}
    for package in packages:
        try:
            version = metadata.version(package)
        except metadata.PackageNotFoundError:
            version = "UNKNOWN"
        versions[package] = version
        lines.append(f"{package}=={version}")
    (CONFIG_DIR / "requirements-lock.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (CONFIG_DIR / "environment.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "packages": versions,
                "seed": SEED,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    start = time.time()
    log("Loading raw and legacy audit tables")
    raw = pd.read_csv(RAW_PATH)
    legacy_features = pd.read_csv(LEGACY_FEATURE_PATH)
    if len(raw) != len(legacy_features):
        raise ValueError("Raw and legacy feature row counts differ")
    raw = raw.copy()
    raw.insert(0, "row_id", np.arange(len(raw)))

    audit = build_canonical_table(raw, legacy_features)
    audit.to_csv(DATA_DIR / "formula_canonicalization_audit.csv", index=False)

    raw_with_key = raw.copy()
    raw_with_key["canonical_key"] = audit["canonical_key_8"]
    raw_with_key["canonical_formula"] = audit["canonical_formula"]
    raw_with_key.to_csv(DATA_DIR / "clean_rows.csv", index=False)

    alias_table = (
        audit.groupby("canonical_key_8")
        .agg(
            canonical_formula=("canonical_formula", "first"),
            reduced_formula=("reduced_formula", "first"),
            n_records=("row_id", "size"),
            n_raw_strings=("raw_formula", "nunique"),
            raw_strings=("raw_formula", lambda values: " || ".join(sorted(set(map(str, values))))),
        )
        .reset_index()
        .rename(columns={"canonical_key_8": "canonical_key"})
    )
    alias_table.to_csv(DATA_DIR / "formula_group_summary.csv", index=False)

    log("Building corrected deterministic composition descriptors")
    feature_artifacts = build_features(raw, audit)
    feature_artifacts.composition.to_csv(DATA_DIR / "features_composition_only.csv", index=False)
    feature_artifacts.context.to_csv(DATA_DIR / "features_composition_context.csv", index=False)
    feature_artifacts.manifest.to_csv(CONFIG_DIR / "feature_manifest.csv", index=False)

    y = pd.to_numeric(raw["Value"], errors="raise").astype(float)
    groups = audit["canonical_key_8"].to_numpy(str)

    fold_assignments = audit[["row_id", "raw_formula", "canonical_key_8"]].copy()
    fold_assignments["loco_group_index"] = pd.factorize(fold_assignments["canonical_key_8"], sort=True)[0]
    fold_assignments.to_csv(DATA_DIR / "canonical_fold_assignments.csv", index=False)

    log("Running secondary record-level CatBoost CV")
    record_oof = evaluate_record_cv(feature_artifacts.context, y, groups)
    record_oof.to_csv(RESULTS_DIR / "oof_record_cv_catboost_context.csv", index=False)

    all_oof = []
    summary_rows = []
    for feature_name, X in [
        ("composition_only", feature_artifacts.composition),
        ("composition_plus_context", feature_artifacts.context),
    ]:
        for model_name, factory in model_factories().items():
            label = f"{model_name}__{feature_name}"
            log(f"Running canonical-composition-held-out evaluation: {label}")
            oof = evaluate_logo(
                X,
                y,
                groups,
                label,
                factory,
                with_distances=(model_name == "catboost" and feature_name == "composition_plus_context"),
            )
            all_oof.append(oof)
            point = metrics(oof["y_true"].to_numpy(), oof["y_pred"].to_numpy())
            balanced = formula_balanced_metrics(
                oof["y_true"].to_numpy(), oof["y_pred"].to_numpy(), groups
            )
            ci = bootstrap_metrics_by_group(oof)
            summary_rows.append(
                {
                    "model": model_name,
                    "feature_set": feature_name,
                    "n_records": len(oof),
                    "n_canonical_compositions": len(np.unique(groups)),
                    **{f"record_{key}": value for key, value in point.items()},
                    **{f"formula_balanced_{key}": value for key, value in balanced.items()},
                    **{f"record_{key}_ci95_low": ci[key][0] for key in ci},
                    **{f"record_{key}_ci95_high": ci[key][1] for key in ci},
                }
            )

    oof_all = pd.concat(all_oof, ignore_index=True)
    oof_all.to_csv(RESULTS_DIR / "oof_canonical_composition_heldout_all_models.csv", index=False)
    model_summary = pd.DataFrame(summary_rows)
    model_summary.to_csv(RESULTS_DIR / "model_baselines_summary.csv", index=False)

    primary_oof = oof_all[
        oof_all["model"] == "catboost__composition_plus_context"
    ].reset_index(drop=True)
    formula_distance, distance_result = distance_inference(primary_oof)
    formula_distance.to_csv(RESULTS_DIR / "distance_error_formula_level.csv", index=False)
    (RESULTS_DIR / "distance_error_inference.json").write_text(
        json.dumps(distance_result, indent=2), encoding="utf-8"
    )

    log("Fitting full corrected CatBoost surrogate")
    full_model = catboost_factory()
    full_model.fit(feature_artifacts.context, y)
    full_model.save_model(MODELS_DIR / "catboost_full_context.cbm")
    joblib.dump(full_model, MODELS_DIR / "catboost_full_context.joblib")
    (CONFIG_DIR / "catboost_params.json").write_text(
        json.dumps(full_model.get_params(), indent=2), encoding="utf-8"
    )
    importance = pd.DataFrame(
        {
            "feature": feature_artifacts.context.columns,
            "importance": full_model.get_feature_importance(),
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(RESULTS_DIR / "catboost_feature_importance.csv", index=False)

    record_point = metrics(record_oof["y_true"].to_numpy(), record_oof["y_pred"].to_numpy())
    record_ci = bootstrap_metrics_by_group(record_oof.assign(abs_error=lambda x: abs(x.y_true - x.y_pred)))
    primary_row = model_summary[
        (model_summary["model"] == "catboost")
        & (model_summary["feature_set"] == "composition_plus_context")
    ].iloc[0]

    data_audit = {
        "raw_rows": int(len(raw)),
        "raw_formula_strings": int(raw["formula"].nunique()),
        "whitespace_normalized_formulas": int(audit["whitespace_normalized_formula"].nunique()),
        "canonical_compositions_decimals_6": int(audit["canonical_key_6"].nunique()),
        "canonical_compositions_decimals_8": int(audit["canonical_key_8"].nunique()),
        "canonical_compositions_decimals_10": int(audit["canonical_key_10"].nunique()),
        "parse_failures": int((~audit["parse_ok"]).sum()),
        "raw_vs_extracted_mismatch_rows": int(audit["raw_vs_extracted_mismatch"].sum()),
        "raw_vs_legacy_feature_mismatch_rows": int(audit["raw_vs_legacy_feature_mismatch"].sum()),
        "legacy_zero_element_vector_rows": int(audit["legacy_zero_element_vector"].sum()),
        "carbon_in_extracted_name_rows": int(audit["carbon_in_extracted_name"].sum()),
        "legacy_is_doped_inconsistent_canonical_groups": int(
            (raw_with_key.groupby("canonical_key")["is_doped"].nunique() > 1).sum()
        ),
        "median_records_per_canonical_composition": float(alias_table["n_records"].median()),
        "maximum_records_per_canonical_composition": int(alias_table["n_records"].max()),
        "composition_only_feature_count": int(feature_artifacts.composition.shape[1]),
        "context_feature_count": int(feature_artifacts.context.shape[1]),
        "constant_context_columns": int(
            (feature_artifacts.context.nunique(dropna=False) <= 1).sum()
        ),
    }
    (RESULTS_DIR / "data_audit_summary.json").write_text(
        json.dumps(data_audit, indent=2), encoding="utf-8"
    )

    manuscript_results = {
        "run_id": "paper_release_v1_canonical_rebuild",
        "data": data_audit,
        "record_cv_catboost_context": {
            **record_point,
            "cluster_bootstrap_ci95": record_ci,
        },
        "canonical_composition_heldout_catboost_context": {
            key.removeprefix("record_"): float(primary_row[key])
            for key in ["record_r2", "record_mae", "record_rmse", "record_spearman"]
        },
        "distance_error_formula_level": distance_result,
        "search_benchmark": "PENDING_SEARCH_SCRIPT",
        "screening": "PENDING_SEARCH_SCRIPT",
    }
    (RESULTS_DIR / "results_for_manuscript.json").write_text(
        json.dumps(manuscript_results, indent=2), encoding="utf-8"
    )

    source_manifest = []
    manifest_sources = (
        sorted(SOURCE_DIR.iterdir())
        if SOURCE_DIR.exists()
        else [RAW_PATH, LEGACY_FEATURE_PATH]
    )
    for path in manifest_sources:
        if path.is_file():
            source_manifest.append(
                {"file": path.name, "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            )
    (DATA_DIR / "source_file_manifest.json").write_text(
        json.dumps(source_manifest, indent=2), encoding="utf-8"
    )
    save_environment()
    log(f"Canonical rebuild completed in {(time.time() - start) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
