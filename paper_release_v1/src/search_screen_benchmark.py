#!/usr/bin/env python3
"""Matched multi-seed search benchmark and transparent post-hoc screening.

The benchmark isolates two factors:

1. proposal mechanism: NSGA-II selection pressure vs independent random search;
2. final prototype coverage floor: on vs off.

All arms share the same prototype definitions, per-prototype objective-evaluation
budget, surrogate, distance discount, Pareto ranking, TopK size, and post-hoc
G1--G4 implementation.  No KNN or chemistry hard gate is applied during search.
"""

from __future__ import annotations

import json
import gzip
import hashlib
import io
import math
import os
import random
import sys
import time
from functools import lru_cache
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from matminer.featurizers.composition import ElementProperty, Stoichiometry
from pymatgen.core import Composition, Element, Species
from scipy.stats import entropy
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

from rebuild_data_and_models import (
    ANIONS,
    CONTEXT_COLUMNS,
    PERIODIC_ELEMENTS,
    TM_SITE,
    canonical_key,
    sanitize,
)


ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "paper_release_v1"
DATA_DIR = RELEASE / "data"
CONFIG_DIR = RELEASE / "config"
RESULTS_DIR = RELEASE / "results"
MODELS_DIR = RELEASE / "models"
SEARCH_DIR = RESULTS_DIR / "search_candidates"
SEARCH_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = list(range(42, 62))
POP_PER_PROTO = 100
N_GENERATIONS = 25
EVALUATIONS_PER_PROTO = POP_PER_PROTO * N_GENERATIONS
TOPK = 500
FLOOR_PER_PROTO = 10
ELITE_FRAC = 0.25
MUT_RATE = 0.85
CROSS_RATE = 0.25
IMMIGRATION_RATE = 0.10
CHECKPOINT_SCHEMA_VERSION = 3

RUNTIME_PACKAGES = [
    "catboost", "matminer", "numpy", "pandas", "pymatgen",
    "scikit-learn", "scipy",
]

FIXED_TEMPERATURE_K = 298.15
FIXED_CURRENT_RATE_C = 0.5
FIXED_CYCLE = 1.0
DISTANCE_DECAY = 0.50
KNN_K = 5

MAX_DOPANT_FRAC = 0.30
DOPANT_FRAC_STEPS = [0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]
LI_EXCESS_RANGE = [-0.10, 0.00, 0.05, 0.10, 0.15, 0.20]
RADIUS_TOL_GENERATOR = 0.25

IONIC_RADII_VI = {
    "Li": 0.76, "Na": 1.02, "K": 1.38, "Mg": 0.72, "Ca": 1.00,
    "Ti": 0.61, "V": 0.54, "Cr": 0.62, "Mn": 0.67, "Fe": 0.65,
    "Co": 0.65, "Ni": 0.69, "Cu": 0.73, "Zn": 0.74, "Al": 0.54,
    "Si": 0.40, "Zr": 0.72, "Ge": 0.53, "Sn": 0.69, "Nb": 0.64,
    "Mo": 0.59, "W": 0.60, "Sc": 0.75, "Ga": 0.62, "In": 0.80,
    "Ru": 0.62, "La": 1.03, "Gd": 0.94, "Y": 0.90, "Ta": 0.64,
    "B": 0.27, "P": 0.38,
}

TM_SET = {
    "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Al", "Mg", "Zr", "Nb", "Mo",
    "W", "Ga", "Sc", "Y", "La", "Gd", "Ru", "Sn", "Ge",
}

PROTO_FAMILY = {
    "Li-rich-NMC": "layered", "NMC-532": "layered", "NMC-622": "layered",
    "NMC-811": "layered", "NCA": "layered", "LiNiO2": "layered", "LiCoO2": "layered",
    "Li2MnO3": "layered_lirich", "LiMn2O4": "spinel", "LNMO-spinel": "spinel",
    "LiFePO4": "olivine", "LiMnPO4": "olivine",
}

LI_TM_WINDOW = {
    "layered": (0.80, 1.50),
    "layered_lirich": (1.50, 2.20),
    "spinel": (0.40, 0.60),
    "olivine": (0.80, 1.20),
}

R6_LAYERED = {
    "Ti": 0.605, "V": 0.580, "Cr": 0.615, "Mn": 0.530, "Fe": 0.645,
    "Co": 0.545, "Ni": 0.560, "Cu": 0.730, "Zn": 0.740,
    "Al": 0.535, "Mg": 0.720, "Zr": 0.720, "Nb": 0.640, "Mo": 0.590,
    "W": 0.600, "Ga": 0.620, "Sc": 0.745, "Y": 0.900, "La": 1.032,
    "Gd": 0.938, "Ru": 0.620, "Sn": 0.690, "Ge": 0.530, "Si": 0.400,
    "B": 0.270, "P": 0.380, "Na": 1.020, "K": 1.380, "Ca": 1.000,
    "O": 1.400, "F": 1.330, "Li": 0.760,
}
R6_OLIVINE = dict(
    R6_LAYERED,
    **{"Fe": 0.780, "Mn": 0.830, "Co": 0.745, "Ni": 0.690, "V": 0.790, "Cr": 0.730, "Ti": 0.860},
)
PROTO_HOST_R = {
    "Li-rich-NMC": 0.540, "NMC-532": 0.580, "NMC-622": 0.595, "NMC-811": 0.580,
    "NCA": 0.570, "LiNiO2": 0.560, "LiCoO2": 0.545,
    "LiMn2O4": 0.590, "LNMO-spinel": 0.570, "Li2MnO3": 0.530,
    "LiFePO4": 0.780, "LiMnPO4": 0.830,
}

REDOX_ELECTRONS = {
    "Ni": 2, "Co": 1, "Mn": 1, "Fe": 1, "V": 2, "Cr": 1, "Cu": 1,
    "Ti": 0, "Al": 0, "Mg": 0, "Zn": 0, "Zr": 0, "Nb": 0, "Mo": 0,
    "W": 0, "Ga": 0, "Sc": 0, "Y": 0, "La": 0, "Gd": 0, "Ru": 0,
    "Sn": 0, "Ge": 0,
}
FARADAY = 96485.0


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def prototypes() -> list[dict[str, object]]:
    return [
        {
            "name": "Li-rich-NMC",
            "base_comp": {"Li": 1.2, "Ni": 0.2, "Mn": 0.6, "O": 2},
            "m_sites": {
                "Ni": {"amount": 0.2, "candidates": ["Co", "Fe", "Ti", "Al", "Mg", "Cr"]},
                "Mn": {"amount": 0.6, "candidates": ["Ti", "Co", "Fe", "V", "Cr"]},
            },
            "li_site_dopants": ["Na"],
        },
        {
            "name": "NMC-532",
            "base_comp": {"Li": 1, "Ni": 0.5, "Mn": 0.3, "Co": 0.2, "O": 2},
            "m_sites": {
                "Ni": {"amount": 0.5, "candidates": ["Al", "Ti", "Fe", "Mg", "Cr", "Zn"]},
                "Co": {"amount": 0.2, "candidates": ["Al", "Ti", "Mn", "Fe"]},
            },
            "li_site_dopants": [],
        },
        {
            "name": "NMC-622",
            "base_comp": {"Li": 1, "Ni": 0.6, "Mn": 0.2, "Co": 0.2, "O": 2},
            "m_sites": {
                "Ni": {"amount": 0.6, "candidates": ["Al", "Ti", "Fe", "Mg", "Cr"]},
                "Co": {"amount": 0.2, "candidates": ["Al", "Ti", "Mn"]},
            },
            "li_site_dopants": [],
        },
        {
            "name": "NMC-811",
            "base_comp": {"Li": 1, "Ni": 0.8, "Mn": 0.1, "Co": 0.1, "O": 2},
            "m_sites": {"Ni": {"amount": 0.8, "candidates": ["Al", "Ti", "Mg", "Fe", "Cr"]}},
            "li_site_dopants": [],
        },
        {
            "name": "NCA",
            "base_comp": {"Li": 1, "Ni": 0.8, "Co": 0.15, "Al": 0.05, "O": 2},
            "m_sites": {
                "Ni": {"amount": 0.8, "candidates": ["Ti", "Mg", "Fe", "Mn", "Cr"]},
                "Co": {"amount": 0.15, "candidates": ["Al", "Ti", "Mn"]},
            },
            "li_site_dopants": [],
        },
        {
            "name": "LiNiO2",
            "base_comp": {"Li": 1, "Ni": 1, "O": 2},
            "m_sites": {"Ni": {"amount": 1, "candidates": ["Al", "Ti", "Co", "Mn", "Fe", "Cr", "Mg", "V"]}},
            "li_site_dopants": [],
        },
        {
            "name": "LiCoO2",
            "base_comp": {"Li": 1, "Co": 1, "O": 2},
            "m_sites": {"Co": {"amount": 1, "candidates": ["Ni", "Mn", "Al", "Fe", "Ti", "Cr", "Mg", "V"]}},
            "li_site_dopants": [],
        },
        {
            "name": "LiMn2O4",
            "base_comp": {"Li": 1, "Mn": 2, "O": 4},
            "m_sites": {"Mn": {"amount": 2, "candidates": ["Ni", "Co", "Fe", "Ti", "Al", "Cr", "V"]}},
            "li_site_dopants": [],
        },
        {
            "name": "LNMO-spinel",
            "base_comp": {"Li": 1, "Ni": 0.5, "Mn": 1.5, "O": 4},
            "m_sites": {
                "Ni": {"amount": 0.5, "candidates": ["Fe", "Co", "Ti", "Cr", "Al"]},
                "Mn": {"amount": 1.5, "candidates": ["Ti", "Fe", "Cr", "V"]},
            },
            "li_site_dopants": [],
        },
        {
            "name": "Li2MnO3",
            "base_comp": {"Li": 2, "Mn": 1, "O": 3},
            "m_sites": {"Mn": {"amount": 1, "candidates": ["Ti", "Fe", "Co", "Ni", "Cr", "V"]}},
            "li_site_dopants": [],
            "anion_variants": [{"replace": "O", "with": "F", "fracs": [0.015, 0.03, 0.045]}],
        },
        {
            "name": "LiFePO4",
            "base_comp": {"Li": 1, "Fe": 1, "P": 1, "O": 4},
            "m_sites": {"Fe": {"amount": 1, "candidates": ["Mn", "Co", "Ni", "V", "Ti", "Mg", "Cr", "Zn"]}},
            "li_site_dopants": ["Na"],
        },
        {
            "name": "LiMnPO4",
            "base_comp": {"Li": 1, "Mn": 1, "P": 1, "O": 4},
            "m_sites": {"Mn": {"amount": 1, "candidates": ["Fe", "Co", "Ni", "V", "Ti", "Mg", "Cr", "Zn"]}},
            "li_site_dopants": ["Na"],
        },
    ]


def radius_compatible(host: str, dopant: str) -> bool:
    return abs(IONIC_RADII_VI.get(host, 0.70) - IONIC_RADII_VI.get(dopant, 0.70)) <= RADIUS_TOL_GENERATOR


def formula_from_dict(values: dict[str, float]) -> str:
    ordered = {key: values[key] for key in sorted(values) if values[key] > 1e-8}
    return Composition(ordered).formula.replace(" ", "")


def legalize_comp(
    values: dict[str, float], proto: dict[str, object]
) -> tuple[dict[str, float], dict[str, float | bool]]:
    """Project every proposal back into the declared prototype parameter bounds.

    Evolutionary mutation is cumulative, whereas independent random proposals are
    drawn directly from the configured bounds.  Without this projection, repeated
    Li tweaks can leave the random-search support and invalidate the matched
    benchmark.  The projection is deliberately chemical-model agnostic: it only
    enforces the same Li-site, substitution, framework-cation, and anion totals
    available to both proposal mechanisms.
    """
    comp = {key: float(values[key]) for key in sorted(values) if float(values[key]) > 1e-10}
    base = {key: float(value) for key, value in proto["base_comp"].items()}

    base_li = base.get("Li", 0.0)
    li_dopants = list(proto.get("li_site_dopants", []))
    for dopant in li_dopants:
        amount = max(0.0, comp.get(dopant, 0.0))
        allowed = 0.10 * base_li
        if amount > allowed:
            comp[dopant] = allowed
            comp["Li"] = comp.get("Li", 0.0) + amount - allowed

    li_total = comp.get("Li", 0.0) + sum(comp.get(item, 0.0) for item in li_dopants)
    li_lower = base_li + min(LI_EXCESS_RANGE)
    li_upper = base_li + max(LI_EXCESS_RANGE)
    target_li_total = min(max(li_total, li_lower), li_upper)
    non_li_site = sum(comp.get(item, 0.0) for item in li_dopants)
    comp["Li"] = max(1e-8, target_li_total - non_li_site)

    m_sites = proto.get("m_sites", {})
    framework_element_set = set(m_sites)
    for info in m_sites.values():
        framework_element_set.update(info["candidates"])
    framework_elements = sorted(framework_element_set)
    framework_target = sum(base.get(element, 0.0) for element in framework_elements)
    framework_total = sum(comp.get(element, 0.0) for element in framework_elements)
    if framework_elements and abs(framework_total - framework_target) > 1e-8:
        host = max(m_sites, key=lambda element: base.get(element, 0.0))
        comp[host] = max(1e-8, comp.get(host, 0.0) + framework_target - framework_total)

    base_framework_elements = set(m_sites)
    nonbase_dopants = sorted(framework_element_set - base_framework_elements)
    dopant_total = sum(comp.get(element, 0.0) for element in nonbase_dopants)
    dopant_limit = MAX_DOPANT_FRAC * framework_target
    if dopant_total > dopant_limit + 1e-10 and dopant_total > 0:
        scale = dopant_limit / dopant_total
        removed = 0.0
        for element in nonbase_dopants:
            before = comp.get(element, 0.0)
            after = before * scale
            comp[element] = after
            removed += before - after
        host = max(m_sites, key=lambda element: base.get(element, 0.0))
        comp[host] = comp.get(host, 0.0) + removed

    anion_elements = sorted(ANIONS)
    anion_target = sum(base.get(element, 0.0) for element in anion_elements)
    anion_total = sum(comp.get(element, 0.0) for element in anion_elements)
    if anion_target > 0 and abs(anion_total - anion_target) > 1e-8:
        host_anion = max((element for element in anion_elements if base.get(element, 0.0) > 0), key=base.get)
        comp[host_anion] = max(1e-8, comp.get(host_anion, 0.0) + anion_target - anion_total)

    comp = {key: round(comp[key], 6) for key in sorted(comp) if comp[key] > 1e-8}
    li_total = comp.get("Li", 0.0) + sum(comp.get(item, 0.0) for item in li_dopants)
    framework_total = sum(comp.get(element, 0.0) for element in framework_elements)
    anion_total = sum(comp.get(element, 0.0) for element in anion_elements)
    dopant_total = sum(comp.get(element, 0.0) for element in nonbase_dopants)
    bounds_ok = bool(
        li_lower - 1e-6 <= li_total <= li_upper + 1e-6
        and abs(framework_total - framework_target) <= 5e-6
        and abs(anion_total - anion_target) <= 5e-6
        and dopant_total <= dopant_limit + 5e-6
    )
    if not bounds_ok:
        raise AssertionError(f"Candidate escaped declared bounds for {proto['name']}: {comp}")
    audit = {
        "li_site_total": li_total,
        "li_site_lower": li_lower,
        "li_site_upper": li_upper,
        "framework_cation_total": framework_total,
        "framework_cation_target": framework_target,
        "nonbase_dopant_total": dopant_total,
        "nonbase_dopant_limit": dopant_limit,
        "anion_total": anion_total,
        "anion_target": anion_target,
        "parameter_bounds_ok": bounds_ok,
    }
    return comp, audit


def candidate_from_comp(comp: dict[str, float], proto: dict[str, object]) -> dict[str, object]:
    comp, bounds_audit = legalize_comp(comp, proto)
    base_key = canonical_key(Composition(proto["base_comp"]))
    comp_obj = Composition(comp)
    return {
        "comp": comp,
        "Formula": formula_from_dict(comp),
        "canonical_key": canonical_key(comp_obj),
        "proto_name": str(proto["name"]),
        "is_doped": int(canonical_key(comp_obj) != base_key),
        **bounds_audit,
    }


def generate_one_candidate(proto: dict[str, object], rng: random.Random) -> dict[str, object]:
    comp = dict(proto["base_comp"])
    for host, info in proto.get("m_sites", {}).items():
        total = float(info["amount"])
        choices = [item for item in info["candidates"] if radius_compatible(host, item)]
        if not choices:
            continue
        roll = rng.random()
        if roll < 0.30:
            continue
        if roll < 0.80:
            dopant = rng.choice(choices)
            fraction = min(rng.choice(DOPANT_FRAC_STEPS), MAX_DOPANT_FRAC)
            amount = round(total * fraction, 6)
            comp[host] = round(total - amount, 6)
            comp[dopant] = round(comp.get(dopant, 0.0) + amount, 6)
        else:
            first, second = rng.sample(choices, 2) if len(choices) >= 2 else (choices[0], choices[0])
            f1 = rng.choice(DOPANT_FRAC_STEPS[:5])
            f2 = min(rng.choice(DOPANT_FRAC_STEPS[:5]), MAX_DOPANT_FRAC - f1)
            a1, a2 = round(total * f1, 6), round(total * f2, 6)
            comp[host] = round(total - a1 - a2, 6)
            comp[first] = round(comp.get(first, 0.0) + a1, 6)
            comp[second] = round(comp.get(second, 0.0) + a2, 6)

    li_dopants = proto.get("li_site_dopants", [])
    if li_dopants and rng.random() < 0.15:
        dopant = rng.choice(li_dopants)
        fraction = rng.choice([0.02, 0.03, 0.05, 0.07, 0.10])
        li_amount = comp.get("Li", 1.0)
        amount = round(li_amount * fraction, 6)
        comp["Li"] = round(li_amount - amount, 6)
        comp[dopant] = round(comp.get(dopant, 0.0) + amount, 6)
    if rng.random() < 0.30:
        comp["Li"] = round(comp.get("Li", 1.0) + rng.choice(LI_EXCESS_RANGE), 6)
    for variant in proto.get("anion_variants", []):
        if rng.random() < 0.40 and variant["replace"] in comp:
            original = comp[variant["replace"]]
            amount = round(original * rng.choice(variant["fracs"]), 6)
            comp[variant["replace"]] = round(original - amount, 6)
            comp[variant["with"]] = round(comp.get(variant["with"], 0.0) + amount, 6)
    return candidate_from_comp(comp, proto)


def mutate(candidate: dict[str, object], proto: dict[str, object], rng: random.Random) -> dict[str, object]:
    comp = dict(candidate["comp"])
    operation = rng.choice(["tweak_li", "replace_element", "tweak_fraction"])
    m_sites = proto.get("m_sites", {})

    if operation == "tweak_li":
        comp["Li"] = max(0.5, round(comp.get("Li", 1.0) + rng.choice([-0.10, -0.05, 0.05, 0.10, 0.15]), 6))
    elif m_sites:
        host = rng.choice(list(m_sites))
        info = m_sites[host]
        choices = [item for item in info["candidates"] if radius_compatible(host, item)]
        base_elements = set(proto["base_comp"])
        dopants = [item for item in choices if item in comp and item not in base_elements]
        if operation == "replace_element" and dopants:
            old = rng.choice(dopants)
            new = rng.choice([item for item in choices if item != old] or choices)
            amount = comp.pop(old)
            comp[new] = round(comp.get(new, 0.0) + amount, 6)
        elif operation == "tweak_fraction" and dopants:
            dopant = rng.choice(dopants)
            total = float(info["amount"])
            delta = total * rng.choice([-0.05, -0.02, 0.02, 0.05])
            old_amount = comp[dopant]
            new_amount = min(max(0.0, old_amount + delta), total * MAX_DOPANT_FRAC)
            comp[dopant] = round(new_amount, 6)
            comp[host] = round(comp.get(host, 0.0) + old_amount - new_amount, 6)
    return candidate_from_comp(comp, proto)


def crossover(
    first: dict[str, object], second: dict[str, object], proto: dict[str, object], rng: random.Random
) -> dict[str, object]:
    comp = {}
    weight = rng.uniform(0.3, 0.7)
    for element in sorted(set(first["comp"]) | set(second["comp"])):
        comp[element] = round(
            float(first["comp"].get(element, 0.0)) * weight
            + float(second["comp"].get(element, 0.0)) * (1.0 - weight),
            6,
        )
    return candidate_from_comp(comp, proto)


class CandidateEvaluator:
    def __init__(self) -> None:
        self.model: CatBoostRegressor = joblib.load(MODELS_DIR / "catboost_full_context.joblib")
        self.training_comp = pd.read_csv(DATA_DIR / "features_composition_only.csv")
        self.training_context = pd.read_csv(DATA_DIR / "features_composition_context.csv")
        self.clean_rows = pd.read_csv(DATA_DIR / "clean_rows.csv")
        self.feature_columns = list(self.training_context.columns)
        self.comp_columns = list(self.training_comp.columns)
        self.stoich = Stoichiometry()
        self.magpie = ElementProperty.from_preset("magpie")
        self.stoich_labels = [sanitize(f"Stoich_{label}") for label in self.stoich.feature_labels()]
        self.magpie_raw_labels = list(self.magpie.feature_labels())
        self.magpie_label_set = {
            sanitize(f"Magpie_{label}")
            for label in self.magpie_raw_labels
            if "_range_" not in sanitize(f"Magpie_{label}")
        }
        self.cache: dict[str, dict[str, float]] = {}

        grouped = self.training_comp.copy()
        grouped["canonical_key"] = self.clean_rows["canonical_key"]
        unique_train = grouped.groupby("canonical_key", sort=True)[self.comp_columns].median()
        self.unique_train = unique_train
        self.imputer = SimpleImputer(strategy="median")
        train_imputed = self.imputer.fit_transform(unique_train)
        self.nonconstant = np.nanstd(train_imputed, axis=0) > 1e-12
        self.scaler = RobustScaler()
        train_scaled = self.scaler.fit_transform(train_imputed[:, self.nonconstant])
        self.model_nn = NearestNeighbors(n_neighbors=KNN_K + 1, metric="euclidean", n_jobs=1).fit(train_scaled)
        self.train_scaled = train_scaled
        self_distance, _ = self.model_nn.kneighbors(train_scaled)
        self.model_distance_p95 = float(np.percentile(self_distance[:, 1:].mean(axis=1), 95))

        fractions = []
        for key in unique_train.index:
            values = {token.split(":")[0]: float(token.split(":")[1]) for token in key.split("|")}
            fractions.append([values.get(element, 0.0) for element in PERIODIC_ELEMENTS])
        self.train_fraction = np.asarray(fractions, dtype=float)
        self.comp_nn = NearestNeighbors(n_neighbors=KNN_K, metric="manhattan", n_jobs=1).fit(self.train_fraction)

        raw_y = pd.to_numeric(self.clean_rows["Value"], errors="coerce").to_numpy(float)
        self.prediction_clip = [float(np.percentile(raw_y, 0.5)), float(np.percentile(raw_y, 99.5))]

    def _composition_row(self, candidate: dict[str, object]) -> dict[str, float]:
        cache_key = str(candidate["canonical_key"])
        if cache_key in self.cache:
            return self.cache[cache_key]
        comp = Composition(candidate["comp"])
        fractions = comp.fractional_composition.get_el_amt_dict()
        row = {f"elemfrac_{element}": float(fractions.get(element, 0.0)) for element in PERIODIC_ELEMENTS}
        for label, value in zip(self.stoich_labels, self.stoich.featurize(comp)):
            row[label] = float(value)
        for raw_label, value in zip(self.magpie_raw_labels, self.magpie.featurize(comp)):
            label = sanitize(f"Magpie_{raw_label}")
            if label in self.magpie_label_set:
                row[label] = float(value)
        row["anion_fraction_sum"] = float(sum(fractions.get(element, 0.0) for element in ANIONS))
        row["tm_site_fraction_sum"] = float(sum(fractions.get(element, 0.0) for element in TM_SITE))
        self.cache[cache_key] = row
        return row

    def evaluate(self, candidates: list[dict[str, object]]) -> pd.DataFrame:
        comp_rows = [self._composition_row(candidate) for candidate in candidates]
        X_comp = pd.DataFrame(comp_rows).reindex(columns=self.comp_columns)
        X_context = X_comp.copy()
        X_context["temperature_K"] = FIXED_TEMPERATURE_K
        X_context["log1p_current_rate_C"] = math.log1p(FIXED_CURRENT_RATE_C)
        X_context["log1p_cycle_number"] = math.log1p(FIXED_CYCLE)
        X_context["log1p_current_rate_x_cycle"] = math.log1p(FIXED_CURRENT_RATE_C * FIXED_CYCLE)
        X_context = X_context.reindex(columns=self.feature_columns)

        prediction_unclipped = np.asarray(self.model.predict(X_context, thread_count=1), dtype=float)
        prediction = np.clip(prediction_unclipped, *self.prediction_clip)

        comp_imputed = self.imputer.transform(X_comp)
        comp_scaled = self.scaler.transform(comp_imputed[:, self.nonconstant])
        model_distances, _ = self.model_nn.kneighbors(comp_scaled, n_neighbors=KNN_K)
        model_distance = model_distances.mean(axis=1)
        excess = np.maximum(0.0, model_distance - self.model_distance_p95)
        discount = np.exp(-DISTANCE_DECAY * excess)
        discounted = prediction * discount

        fraction = X_comp[[f"elemfrac_{element}" for element in PERIODIC_ELEMENTS]].to_numpy(float)
        novelty_distance, _ = self.comp_nn.kneighbors(fraction, n_neighbors=KNN_K)
        novelty = novelty_distance.mean(axis=1)

        output = pd.DataFrame(
            {
                "proto_name": [candidate["proto_name"] for candidate in candidates],
                "Formula": [candidate["Formula"] for candidate in candidates],
                "canonical_key": [candidate["canonical_key"] for candidate in candidates],
                "is_doped": [candidate["is_doped"] for candidate in candidates],
                "initial_shared_batch": [bool(candidate.get("_initial_shared_batch", False)) for candidate in candidates],
                # Known positive controls are evaluated as observed compositions and
                # are not proposal-space members, so their bound-audit fields are NA.
                "li_site_total": [candidate.get("li_site_total", np.nan) for candidate in candidates],
                "li_site_lower": [candidate.get("li_site_lower", np.nan) for candidate in candidates],
                "li_site_upper": [candidate.get("li_site_upper", np.nan) for candidate in candidates],
                "framework_cation_total": [candidate.get("framework_cation_total", np.nan) for candidate in candidates],
                "framework_cation_target": [candidate.get("framework_cation_target", np.nan) for candidate in candidates],
                "nonbase_dopant_total": [candidate.get("nonbase_dopant_total", np.nan) for candidate in candidates],
                "nonbase_dopant_limit": [candidate.get("nonbase_dopant_limit", np.nan) for candidate in candidates],
                "anion_total": [candidate.get("anion_total", np.nan) for candidate in candidates],
                "anion_target": [candidate.get("anion_target", np.nan) for candidate in candidates],
                "parameter_bounds_ok": [candidate.get("parameter_bounds_ok", np.nan) for candidate in candidates],
                "y_pred_unclipped": prediction_unclipped,
                "y_pred": prediction,
                "model_distance": model_distance,
                "model_distance_p95": self.model_distance_p95,
                "discount": discount,
                "y_discounted": discounted,
                "comp_novelty": novelty,
            }
        )
        return output


class FenwickMax:
    def __init__(self, size: int) -> None:
        self.tree = np.full(size + 1, -1, dtype=int)

    def update(self, index: int, value: int) -> None:
        while index < len(self.tree):
            self.tree[index] = max(self.tree[index], value)
            index += index & -index

    def query(self, index: int) -> int:
        result = -1
        while index > 0:
            result = max(result, int(self.tree[index]))
            index -= index & -index
        return result


def pareto_ranks(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    """O(n log n) exact nondominated ranks for two maximized objectives."""
    f1 = np.asarray(f1, dtype=float)
    f2 = np.asarray(f2, dtype=float)
    levels = sorted(set(f2), reverse=True)
    level_index = {value: idx + 1 for idx, value in enumerate(levels)}
    order = sorted(range(len(f1)), key=lambda idx: (-f1[idx], -f2[idx], idx))
    ranks = np.empty(len(f1), dtype=int)
    fenwick = FenwickMax(len(levels))
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        pair = (f1[order[cursor]], f2[order[cursor]])
        while end < len(order) and (f1[order[end]], f2[order[end]]) == pair:
            end += 1
        group = order[cursor:end]
        rank = fenwick.query(level_index[pair[1]]) + 1
        for idx in group:
            ranks[idx] = rank
        fenwick.update(level_index[pair[1]], rank)
        cursor = end
    return ranks


def crowding_distance(f1: np.ndarray, f2: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    distance = np.zeros(len(f1), dtype=float)
    for rank in np.unique(ranks):
        front = np.flatnonzero(ranks == rank)
        if len(front) <= 2:
            distance[front] = np.inf
            continue
        for objective in [f1, f2]:
            ordered = front[np.argsort(objective[front], kind="mergesort")]
            distance[ordered[[0, -1]]] = np.inf
            span = objective[ordered[-1]] - objective[ordered[0]]
            if span <= 1e-12:
                continue
            for pos in range(1, len(ordered) - 1):
                if np.isfinite(distance[ordered[pos]]):
                    distance[ordered[pos]] += (
                        objective[ordered[pos + 1]] - objective[ordered[pos - 1]]
                    ) / span
    return distance


def rank_pool(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.copy()
    f1 = ranked["y_discounted"].to_numpy(float)
    f2 = ranked["comp_novelty"].to_numpy(float)
    ranked["pareto_rank"] = pareto_ranks(f1, f2)
    ranked["crowding_distance"] = crowding_distance(f1, f2, ranked["pareto_rank"].to_numpy())
    ranked = ranked.sort_values(
        ["pareto_rank", "crowding_distance", "y_discounted", "canonical_key", "proto_name"],
        ascending=[True, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return ranked


def unique_in_order(frame: pd.DataFrame, limit: int, used: set[str] | None = None) -> pd.DataFrame:
    used = set() if used is None else used
    indices = []
    for idx, key in zip(frame.index, frame["canonical_key"]):
        if key in used:
            continue
        used.add(str(key))
        indices.append(idx)
        if len(indices) >= limit:
            break
    return frame.loc[indices].copy()


def select_topk(pool: pd.DataFrame, floor_on: bool) -> pd.DataFrame:
    ranked = rank_pool(pool)
    used: set[str] = set()
    selected = []
    if floor_on:
        for proto_name in sorted(ranked["proto_name"].unique()):
            part = unique_in_order(ranked[ranked["proto_name"] == proto_name], FLOOR_PER_PROTO, used)
            selected.append(part)
    base = pd.concat(selected, ignore_index=False) if selected else ranked.iloc[0:0].copy()
    remaining = unique_in_order(ranked, TOPK - len(base), used)
    topk = pd.concat([base, remaining], ignore_index=True)
    topk = rank_pool(topk).head(TOPK).reset_index(drop=True)
    if len(topk) != TOPK:
        raise RuntimeError(f"TopK selection returned {len(topk)} rows")
    return topk


def shared_initial_population(seed: int) -> dict[str, list[dict[str, object]]]:
    """Return the exact first 100 proposals shared by both matched arms."""
    initial: dict[str, list[dict[str, object]]] = {}
    for proto_index, proto in enumerate(prototypes()):
        rng = random.Random(seed * 1009 + proto_index * 7919 + 17)
        batch = [generate_one_candidate(proto, rng) for _ in range(POP_PER_PROTO)]
        for candidate in batch:
            candidate["_initial_shared_batch"] = True
        initial[str(proto["name"])] = batch
    return initial


def clone_candidates(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    return [{**candidate, "comp": dict(candidate["comp"])} for candidate in candidates]


def tournament_index(ranked: pd.DataFrame, rng: random.Random) -> int:
    """Binary NSGA-II tournament: rank, crowding, objective, stable key."""
    first, second = rng.sample(range(len(ranked)), 2)
    left = ranked.iloc[first]
    right = ranked.iloc[second]
    left_key = (
        int(left["pareto_rank"]),
        -float(left["crowding_distance"]),
        -float(left["y_discounted"]),
        str(left["canonical_key"]),
    )
    right_key = (
        int(right["pareto_rank"]),
        -float(right["crowding_distance"]),
        -float(right["y_discounted"]),
        str(right["canonical_key"]),
    )
    return first if left_key <= right_key else second


def evaluated_population(
    candidates: list[dict[str, object]], evaluator: CandidateEvaluator
) -> pd.DataFrame:
    frame = evaluator.evaluate(candidates)
    frame["_candidate_index"] = np.arange(len(candidates), dtype=int)
    return rank_pool(frame)


def nsga_proposal(
    seed: int,
    evaluator: CandidateEvaluator,
    initial: dict[str, list[dict[str, object]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Canonical elitist NSGA-II under an exact 2,500-evaluation budget/prototype.

    Generation 1 evaluates the shared initial batch.  Each later generation
    evaluates 100 offspring, merges parents and offspring, and retains the best
    100 by nondominated rank and crowding distance.  Thus every prototype uses
    exactly 100 + 24 x 100 objective evaluations.
    """
    proto_list = prototypes()
    populations = {
        str(proto["name"]): clone_candidates(initial[str(proto["name"])])
        for proto in proto_list
    }
    scores: dict[str, pd.DataFrame] = {}
    all_rows: list[pd.DataFrame] = []
    history: list[dict[str, object]] = []

    # Evaluate the common initial batch in one call, then split by prototype.
    flat_initial = [
        candidate
        for proto in proto_list
        for candidate in populations[str(proto["name"])]
    ]
    initial_eval = evaluator.evaluate(flat_initial)
    initial_eval["generation"] = 1
    initial_eval["initial_shared_batch"] = True
    initial_eval["seed"] = seed
    initial_eval["proposal"] = "nsga2"
    all_rows.append(initial_eval.copy())
    offset = 0
    for proto in proto_list:
        name = str(proto["name"])
        part = initial_eval.iloc[offset : offset + POP_PER_PROTO].copy().reset_index(drop=True)
        part["_candidate_index"] = np.arange(POP_PER_PROTO, dtype=int)
        scores[name] = rank_pool(part)
        offset += POP_PER_PROTO

    for generation in range(1, N_GENERATIONS + 1):
        if generation > 1:
            offspring_by_proto: dict[str, list[dict[str, object]]] = {}
            for proto_index, proto in enumerate(proto_list):
                name = str(proto["name"])
                rng = random.Random(seed * 1000003 + proto_index * 10007 + generation * 101)
                ranked = scores[name]
                parents = populations[name]
                offspring: list[dict[str, object]] = []
                while len(offspring) < POP_PER_PROTO:
                    if rng.random() < IMMIGRATION_RATE:
                        child = generate_one_candidate(proto, rng)
                    else:
                        first = parents[int(ranked.iloc[tournament_index(ranked, rng)]["_candidate_index"])]
                        if rng.random() < CROSS_RATE:
                            second = parents[int(ranked.iloc[tournament_index(ranked, rng)]["_candidate_index"])]
                            child = crossover(first, second, proto, rng)
                        else:
                            child = {**first, "comp": dict(first["comp"])}
                        if rng.random() < MUT_RATE:
                            child = mutate(child, proto, rng)
                    child.pop("_initial_shared_batch", None)
                    offspring.append(child)
                offspring_by_proto[name] = offspring

            flat_offspring = [
                candidate
                for proto in proto_list
                for candidate in offspring_by_proto[str(proto["name"])]
            ]
            offspring_eval = evaluator.evaluate(flat_offspring)
            offspring_eval["generation"] = generation
            offspring_eval["initial_shared_batch"] = False
            offspring_eval["seed"] = seed
            offspring_eval["proposal"] = "nsga2"
            all_rows.append(offspring_eval.copy())

            offset = 0
            new_populations: dict[str, list[dict[str, object]]] = {}
            new_scores: dict[str, pd.DataFrame] = {}
            for proto in proto_list:
                name = str(proto["name"])
                children = offspring_by_proto[name]
                child_scores = offspring_eval.iloc[offset : offset + POP_PER_PROTO].copy().reset_index(drop=True)
                offset += POP_PER_PROTO
                union_candidates = populations[name] + children
                parent_scores = scores[name].drop(columns=["pareto_rank", "crowding_distance"], errors="ignore").copy()
                parent_scores["_candidate_index"] = np.arange(POP_PER_PROTO, dtype=int)
                child_scores["_candidate_index"] = np.arange(POP_PER_PROTO, 2 * POP_PER_PROTO, dtype=int)
                union_scores = pd.concat([parent_scores, child_scores], ignore_index=True)
                selected = rank_pool(union_scores).head(POP_PER_PROTO).copy()
                chosen = selected["_candidate_index"].astype(int).to_list()
                new_populations[name] = [union_candidates[index] for index in chosen]
                selected["_candidate_index"] = np.arange(POP_PER_PROTO, dtype=int)
                new_scores[name] = selected.reset_index(drop=True)
            populations = new_populations
            scores = new_scores

        current_scores = pd.concat(
            [scores[str(proto["name"])].drop(columns=["_candidate_index"], errors="ignore") for proto in proto_list],
            ignore_index=True,
        )
        top50 = rank_pool(current_scores).head(50)
        history.append(
            {
                "seed": seed,
                "proposal": "nsga2",
                "generation": generation,
                "evaluations": generation * POP_PER_PROTO * len(proto_list),
                "top50_median_y_pred": float(top50["y_pred"].median()),
                "top50_median_y_discounted": float(top50["y_discounted"].median()),
                "top50_median_comp_novelty": float(top50["comp_novelty"].median()),
            }
        )
        log(f"Seed {seed}: NSGA-II generation {generation}/{N_GENERATIONS}")

    result = pd.concat(all_rows, ignore_index=True)
    if len(result) != EVALUATIONS_PER_PROTO * len(proto_list):
        raise AssertionError("NSGA-II objective-evaluation budget mismatch")
    return result, pd.DataFrame(history)


def random_proposal(
    seed: int,
    evaluator: CandidateEvaluator,
    initial: dict[str, list[dict[str, object]]],
) -> pd.DataFrame:
    """Independent random search with the same budget and the same first batch."""
    rows: list[pd.DataFrame] = []
    proto_list = prototypes()
    chunk: list[dict[str, object]] = []
    for proto_index, proto in enumerate(proto_list):
        name = str(proto["name"])
        shared = clone_candidates(initial[name])
        chunk.extend(shared)
        rng = random.Random(seed * 1000033 + proto_index * 10009 + 99991)
        for _ in range(EVALUATIONS_PER_PROTO - POP_PER_PROTO):
            chunk.append(generate_one_candidate(proto, rng))
            if len(chunk) >= 2000:
                rows.append(evaluator.evaluate(chunk))
                chunk = []
    if chunk:
        rows.append(evaluator.evaluate(chunk))
    frame = pd.concat(rows, ignore_index=True)
    frame["generation"] = 0
    frame["seed"] = seed
    frame["proposal"] = "random"
    if len(frame) != EVALUATIONS_PER_PROTO * len(proto_list):
        raise AssertionError("Random-search objective-evaluation budget mismatch")
    return frame


_OXI: dict[str, list[int]] = {}
_EN: dict[str, float] = {}


def oxidation_states(symbol: str) -> list[int]:
    if symbol not in _OXI:
        element = Element(symbol)
        states = list(element.icsd_oxidation_states) or list(element.common_oxidation_states)
        _OXI[symbol] = [int(value) for value in states]
    return _OXI[symbol]


def electronegativity(symbol: str) -> float:
    if symbol not in _EN:
        value = Element(symbol).X
        _EN[symbol] = float(value) if value is not None else 0.0
    return _EN[symbol]


@lru_cache(maxsize=None)
def gate1_ionicity(formula: str, tolerance: float = 0.10) -> tuple[bool, float, bool]:
    """Mixed-valence charge feasibility plus a Pauling ordering sanity check.

    A brute-force Cartesian product of integer oxidation states makes complex
    doped compositions fail merely because an arbitrary combination cap is
    exceeded.  Here, each element may take any average state in the convex hull
    of its allowed same-sign oxidation states, which is the appropriate
    composition-level relaxation for mixed valence.  Charge neutrality is then
    a one-dimensional interval-feasibility test.  Pauling ordering is evaluated
    independently between cationic and anionic species.
    """
    comp = Composition(formula).get_el_amt_dict()
    charge_min = 0.0
    charge_max = 0.0
    cations: list[str] = []
    anions: list[str] = []
    for element in sorted(comp):
        states = oxidation_states(element)
        allowed = [state for state in states if state < 0] if element in ANIONS else [state for state in states if state > 0]
        if not allowed:
            return False, np.nan, False
        amount = float(comp[element])
        charge_min += amount * min(allowed)
        charge_max += amount * max(allowed)
        (anions if element in ANIONS else cations).append(element)

    if charge_min <= 0.0 <= charge_max:
        minimum_abs_charge = 0.0
    else:
        minimum_abs_charge = min(abs(charge_min), abs(charge_max))
    charge_feasible = minimum_abs_charge < tolerance
    pauling_pass = bool(
        cations
        and anions
        and all(electronegativity(cation) <= electronegativity(anion) for cation in cations for anion in anions)
    )
    return bool(charge_feasible and pauling_pass), float(minimum_abs_charge), pauling_pass


@lru_cache(maxsize=None)
def gate2_li_tm(formula: str, proto_name: str, width_scale: float = 1.0) -> tuple[bool, float]:
    comp = Composition(formula).get_el_amt_dict()
    tm_total = sum(value for element, value in comp.items() if element in TM_SET)
    ratio = float(comp.get("Li", 0.0) / tm_total) if tm_total > 0 else np.nan
    family = PROTO_FAMILY.get(proto_name)
    if family is None or not np.isfinite(ratio):
        return False, ratio
    low, high = LI_TM_WINDOW[family]
    center = (low + high) / 2
    half = (high - low) / 2 * width_scale
    return bool(center - half <= ratio <= center + half), ratio


@lru_cache(maxsize=None)
def radius_mismatch_metrics(formula: str, proto_name: str) -> tuple[float, float]:
    """Return concentration-weighted RMS and legacy maximum radius mismatch.

    The weighted RMS is the primary G3 descriptor because it preserves the
    concentration of each framework cation.  The former maximum mismatch is
    retained only as an exported diagnostic.
    """
    comp = Composition(formula).get_el_amt_dict()
    host = PROTO_HOST_R.get(proto_name)
    if host is None:
        return np.nan, np.nan
    radii = R6_OLIVINE if PROTO_FAMILY.get(proto_name) == "olivine" else R6_LAYERED
    weighted_square = 0.0
    total = 0.0
    maximum = 0.0
    for element, amount in comp.items():
        if element not in TM_SET or amount <= 0 or element not in radii:
            continue
        relative = abs(radii[element] - host) / host
        weighted_square += float(amount) * relative**2
        total += float(amount)
        maximum = max(maximum, relative)
    if total <= 0:
        return np.nan, np.nan
    return float(math.sqrt(weighted_square / total)), float(maximum)


@lru_cache(maxsize=None)
def redox_budget(formula: str) -> tuple[float, float, float]:
    comp = Composition(formula)
    lithium = float(comp.get("Li", 0.0))
    weight = float(comp.weight)
    redox_electrons = sum(float(comp.get(element, 0.0)) * REDOX_ELECTRONS.get(element, 0) for element in sorted(TM_SET))
    if lithium <= 0 or weight <= 0:
        return 0.0, 0.0, redox_electrons
    theoretical = lithium * FARADAY / (3.6 * weight)
    cationic = min(lithium, redox_electrons) * FARADAY / (3.6 * weight)
    return float(theoretical), float(cationic), float(redox_electrons)


@lru_cache(maxsize=None)
def jt_proxy(formula: str, proto_name: str) -> float:
    comp = Composition(formula).get_el_amt_dict()
    tm_total = sum(comp.get(element, 0.0) for element in sorted(TM_SET))
    if tm_total <= 0:
        return 0.0
    active = comp.get("Cu", 0.0)
    if proto_name == "LiMn2O4":
        active += 0.5 * comp.get("Mn", 0.0)
    return float(active / tm_total)


def screen(frame: pd.DataFrame, radius_threshold: float) -> pd.DataFrame:
    output = frame.copy()
    g1_cache: dict[str, tuple[bool, float, bool]] = {}
    g2_cache: dict[tuple[str, str], tuple[bool, float]] = {}
    radius_cache: dict[tuple[str, str], tuple[float, float]] = {}
    redox_cache: dict[str, tuple[float, float, float]] = {}
    jt_cache: dict[tuple[str, str], float] = {}
    for row in output.itertuples(index=False):
        key = str(row.canonical_key)
        proto_key = (key, str(row.proto_name))
        g1_cache.setdefault(key, gate1_ionicity(str(row.Formula)))
        g2_cache.setdefault(proto_key, gate2_li_tm(str(row.Formula), str(row.proto_name)))
        radius_cache.setdefault(proto_key, radius_mismatch_metrics(str(row.Formula), str(row.proto_name)))
        redox_cache.setdefault(key, redox_budget(str(row.Formula)))
        jt_cache.setdefault(proto_key, jt_proxy(str(row.Formula), str(row.proto_name)))

    output["g1_pass"] = [g1_cache[str(key)][0] for key in output["canonical_key"]]
    output["g1_min_abs_charge"] = [g1_cache[str(key)][1] for key in output["canonical_key"]]
    output["g1_pauling_pass"] = [g1_cache[str(key)][2] for key in output["canonical_key"]]
    output["g2_pass"] = [g2_cache[(str(key), str(proto))][0] for key, proto in zip(output["canonical_key"], output["proto_name"])]
    output["li_tm_ratio"] = [g2_cache[(str(key), str(proto))][1] for key, proto in zip(output["canonical_key"], output["proto_name"])]
    output["radius_mismatch"] = [radius_cache[(str(key), str(proto))][0] for key, proto in zip(output["canonical_key"], output["proto_name"])]
    output["radius_mismatch_max_diagnostic"] = [radius_cache[(str(key), str(proto))][1] for key, proto in zip(output["canonical_key"], output["proto_name"])]
    output["g3_pass"] = output["radius_mismatch"].le(radius_threshold)
    output["q_theoretical"] = [redox_cache[str(key)][0] for key in output["canonical_key"]]
    output["q_cationic_redox"] = [redox_cache[str(key)][1] for key in output["canonical_key"]]
    output["n_redox_electrons"] = [redox_cache[str(key)][2] for key in output["canonical_key"]]
    output["g4_pass"] = pd.to_numeric(output["y_pred"], errors="coerce").le(output["q_cationic_redox"])
    output["jt_fraction_proxy"] = [jt_cache[(str(key), str(proto))] for key, proto in zip(output["canonical_key"], output["proto_name"])]

    output["pass_g1"] = output["g1_pass"]
    output["pass_g1_g2"] = output["pass_g1"] & output["g2_pass"]
    output["pass_g1_g2_g3"] = output["pass_g1_g2"] & output["g3_pass"]
    output["pass_g1_g2_g3_g4"] = output["pass_g1_g2_g3"] & output["g4_pass"]
    conditions = [
        ~output["g1_pass"],
        output["g1_pass"] & ~output["g2_pass"],
        output["pass_g1_g2"] & ~output["g3_pass"],
        output["pass_g1_g2_g3"] & ~output["g4_pass"],
    ]
    output["first_failure"] = np.select(conditions, ["G1", "G2", "G3", "G4"], default="Pass")
    return output


def hypervolume_2d(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    points = frame[["y_discounted", "comp_novelty"]].dropna().to_numpy(float)
    points = points[(points[:, 0] > 0) & (points[:, 1] > 0)]
    if not len(points):
        return 0.0
    ranks = pareto_ranks(points[:, 0], points[:, 1])
    front = points[ranks == 0]
    front = front[np.argsort(front[:, 0])]
    area = 0.0
    previous_x = 0.0
    for x_value, y_value in front:
        area += max(0.0, x_value - previous_x) * max(0.0, y_value)
        previous_x = max(previous_x, x_value)
    return float(area)


def arm_summary(topk: pd.DataFrame, screened: pd.DataFrame, seed: int, proposal: str, floor_on: bool) -> dict[str, object]:
    counts = topk["proto_name"].value_counts()
    probabilities = counts.to_numpy(float) / len(topk)
    survivors = screened[screened["pass_g1_g2_g3_g4"]]
    return {
        "seed": seed,
        "proposal": proposal,
        "floor": "on" if floor_on else "off",
        "arm": f"{proposal}__floor_{'on' if floor_on else 'off'}",
        "topk_n": int(len(topk)),
        "unique_canonical_n": int(topk["canonical_key"].nunique()),
        "parameter_bounds_violations": int((~topk["parameter_bounds_ok"].astype(bool)).sum()),
        "prototype_coverage": int(topk["proto_name"].nunique()),
        "prototype_entropy": float(entropy(probabilities)),
        "y_pred_mean": float(topk["y_pred"].mean()),
        "y_pred_median": float(topk["y_pred"].median()),
        "y_discounted_mean": float(topk["y_discounted"].mean()),
        "y_discounted_median": float(topk["y_discounted"].median()),
        "model_distance_mean": float(topk["model_distance"].mean()),
        "model_distance_median": float(topk["model_distance"].median()),
        "comp_novelty_mean": float(topk["comp_novelty"].mean()),
        "comp_novelty_median": float(topk["comp_novelty"].median()),
        "topk_hypervolume": hypervolume_2d(topk),
        "g1_survivors": int(screened["pass_g1"].sum()),
        "g2_survivors": int(screened["pass_g1_g2"].sum()),
        "g3_primary_p90_survivors": int(screened["g3_p90_pass"].sum()) if "g3_p90_pass" in screened else int(screened["pass_g1_g2_g3"].sum()),
        "g3_companion_p95_survivors": int(screened["g3_p95_pass"].sum()) if "g3_p95_pass" in screened else np.nan,
        "g3_strict_survivors": int(screened["pass_g1_g2_g3"].sum()),
        "g4_survivors": int(screened["pass_g1_g2_g3_g4"].sum()),
        "feasible_hypervolume": hypervolume_2d(survivors),
    }


def assign_known_prototype(formula: str) -> tuple[str | None, str]:
    comp = Composition(formula).get_el_amt_dict()
    lithium = comp.get("Li", 0.0)
    oxygen = comp.get("O", 0.0)
    phosphorus = comp.get("P", 0.0)
    silicon = comp.get("Si", 0.0)
    tm_total = sum(comp.get(element, 0.0) for element in sorted(TM_SET))

    if phosphorus > 0:
        if abs(lithium - 1) <= 0.25 and abs(phosphorus - 1) <= 0.25 and abs(oxygen - 4) <= 0.5 and 0.7 <= tm_total <= 1.3:
            mn = comp.get("Mn", 0.0)
            fe = comp.get("Fe", 0.0)
            return ("LiMnPO4" if mn > fe else "LiFePO4"), "olivine_stoichiometry"
        return None, "phosphate_outside_olivine_template"
    if silicon > 0 or comp.get("B", 0.0) > 0 or (comp.get("S", 0.0) > 0 and oxygen == 0):
        return None, "chemistry_outside_12_templates"

    anion_total = oxygen + comp.get("F", 0.0)
    if 2.5 <= anion_total <= 3.5 and 1.6 <= lithium <= 2.3 and 0.7 <= tm_total <= 1.3:
        return "Li2MnO3", "li2mno3_like"
    if 3.5 <= anion_total <= 4.5 and 0.7 <= lithium <= 1.3 and 1.6 <= tm_total <= 2.3:
        ni_fraction = comp.get("Ni", 0.0) / tm_total if tm_total else 0
        mn_fraction = comp.get("Mn", 0.0) / tm_total if tm_total else 0
        if ni_fraction >= 0.15 and mn_fraction >= 0.45:
            return "LNMO-spinel", "lnmo_spinel_like"
        return "LiMn2O4", "spinel_like"
    if 1.7 <= anion_total <= 2.3 and 0.75 <= lithium <= 1.5 and 0.7 <= tm_total <= 1.3:
        if lithium > 1.05 and comp.get("Mn", 0.0) > 0:
            return "Li-rich-NMC", "li_rich_layered_like"
        if comp.get("Co", 0.0) / tm_total >= 0.70:
            return "LiCoO2", "cobalt_dominant_layered"
        if comp.get("Ni", 0.0) / tm_total >= 0.90:
            return "LiNiO2", "nickel_dominant_layered"
        if comp.get("Al", 0.0) > 0 and comp.get("Ni", 0.0) / tm_total >= 0.60:
            return "NCA", "nca_like"
        vector = np.array([comp.get("Ni", 0.0), comp.get("Mn", 0.0), comp.get("Co", 0.0)]) / tm_total
        targets = {
            "NMC-532": np.array([0.5, 0.3, 0.2]),
            "NMC-622": np.array([0.6, 0.2, 0.2]),
            "NMC-811": np.array([0.8, 0.1, 0.1]),
        }
        return min(targets, key=lambda name: float(np.linalg.norm(vector - targets[name]))), "nearest_nmc_ratio"
    return None, "outside_template_stoichiometry"


def build_known_controls(evaluator: CandidateEvaluator) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = pd.read_csv(DATA_DIR / "clean_rows.csv")
    group = (
        clean.groupby("canonical_key", as_index=False)
        .agg(
            Formula=("formula", lambda values: sorted(set(map(str, values)))[0].strip()),
            canonical_formula=("canonical_formula", "first"),
            n_records=("row_id", "size"),
            is_doped=("is_doped", "max"),
        )
        .sort_values("canonical_key")
    )
    assignments = group["Formula"].apply(assign_known_prototype)
    group["proto_name"] = [item[0] for item in assignments]
    group["prototype_assignment_rule"] = [item[1] for item in assignments]
    applicable = group[group["proto_name"].notna()].copy()
    candidates = []
    for row in applicable.itertuples(index=False):
        comp = Composition(row.Formula)
        candidates.append(
            {
                "comp": comp.get_el_amt_dict(),
                "Formula": comp.formula.replace(" ", ""),
                "canonical_key": row.canonical_key,
                "proto_name": row.proto_name,
                "is_doped": int(row.is_doped),
            }
        )
    evaluated = evaluator.evaluate(candidates)
    evaluated = evaluated.merge(
        applicable[["canonical_key", "n_records", "prototype_assignment_rule"]],
        on="canonical_key",
        how="left",
    )
    # Only G1--G2 and the radius descriptor are needed before calibration;
    # infinity prevents an accidental legacy cutoff from entering this stage.
    return group, screen(evaluated, radius_threshold=float("inf"))


def summarize_seed_pairs(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "y_pred_mean", "y_discounted_mean", "model_distance_mean", "comp_novelty_mean",
        "prototype_coverage", "prototype_entropy", "g4_survivors", "topk_hypervolume", "feasible_hypervolume",
    ]
    rows = []
    comparisons = [
        ("nsga2__floor_on", "random__floor_on", "proposal_effect_floor_on"),
        ("nsga2__floor_off", "random__floor_off", "proposal_effect_floor_off"),
        ("nsga2__floor_on", "nsga2__floor_off", "floor_effect_nsga2"),
        ("random__floor_on", "random__floor_off", "floor_effect_random"),
    ]
    for left, right, label in comparisons:
        left_frame = summary[summary["arm"] == left].set_index("seed")
        right_frame = summary[summary["arm"] == right].set_index("seed")
        for metric in metrics:
            delta = left_frame[metric] - right_frame[metric]
            finite = delta.dropna().to_numpy(float)
            if len(finite):
                rng = np.random.default_rng(420000 + len(rows))
                draws = rng.choice(finite, size=(20000, len(finite)), replace=True).mean(axis=1)
                ci_low, ci_high = np.percentile(draws, [2.5, 97.5])
            else:
                ci_low, ci_high = np.nan, np.nan
            rows.append(
                {
                    "comparison": label,
                    "left_arm": left,
                    "right_arm": right,
                    "metric": metric,
                    "mean_paired_delta": float(delta.mean()),
                    "sd_paired_delta": float(delta.std(ddof=1)),
                    "n_seeds": int(delta.notna().sum()),
                    "n_positive": int((delta > 0).sum()),
                    "paired_bootstrap_ci95_low": float(ci_low),
                    "paired_bootstrap_ci95_high": float(ci_high),
                    "paired_bootstrap_repetitions": 20000,
                    "per_seed_deltas": "|".join(f"{seed}:{value:.8g}" for seed, value in delta.items()),
                }
            )
    return pd.DataFrame(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path, *, gzip_output: bool = False) -> None:
    """Write a round-trip-stable CSV and replace the destination atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".staging")
    csv_bytes = frame.to_csv(
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    ).encode("utf-8")
    payload = gzip.compress(csv_bytes, compresslevel=6, mtime=0) if gzip_output else csv_bytes
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_checkpoint_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def runtime_fingerprint() -> dict[str, object]:
    """Versions that can change featurization, prediction, or ranking bytes."""
    versions = {}
    for package in RUNTIME_PACKAGES:
        try:
            versions[package] = package_metadata.version(package)
        except package_metadata.PackageNotFoundError:
            versions[package] = "UNKNOWN"
    payload = {"python": sys.version.split()[0], "packages": versions}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def seed_artifacts(seed: int) -> dict[str, Path]:
    return {
        "nsga_pool": SEARCH_DIR / f"nsga2_seed{seed}_candidate_pool.csv.gz",
        "random_pool": SEARCH_DIR / f"random_seed{seed}_candidate_pool.csv.gz",
        "history": SEARCH_DIR / f"nsga2_seed{seed}_generation_history.csv",
        "summary": SEARCH_DIR / f"seed{seed}_arm_summary.csv",
        "threshold": SEARCH_DIR / f"seed{seed}_radius_threshold_sensitivity.csv",
        "checkpoint": SEARCH_DIR / f"seed{seed}_checkpoint.json",
    }


def checkpoint_is_valid(seed: int, model_sha256: str, runtime_sha256: str) -> bool:
    paths = seed_artifacts(seed)
    if not paths["checkpoint"].exists():
        return False
    try:
        metadata = json.loads(paths["checkpoint"].read_text(encoding="utf-8"))
    except Exception:
        return False
    required = ["nsga_pool", "random_pool", "history", "summary", "threshold"]
    if not (
        metadata.get("schema_version") == CHECKPOINT_SCHEMA_VERSION
        and metadata.get("seed") == seed
        and metadata.get("model_sha256") == model_sha256
        and metadata.get("runtime_sha256") == runtime_sha256
        and metadata.get("algorithm") == "canonical_elitist_nsga2_parent_offspring_union"
        and metadata.get("g3_descriptor") == "concentration_weighted_rms_radius_mismatch"
        and all(paths[name].exists() for name in required)
    ):
        return False
    hashes = metadata.get("sha256", {})
    return all(hashes.get(name) == sha256(paths[name]) for name in required)


def write_seed_checkpoint(seed: int, model_sha256: str, runtime_sha256: str) -> None:
    paths = seed_artifacts(seed)
    required = ["nsga_pool", "random_pool", "history", "summary", "threshold"]
    metadata = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "seed": seed,
        "model_sha256": model_sha256,
        "runtime_sha256": runtime_sha256,
        "algorithm": "canonical_elitist_nsga2_parent_offspring_union",
        "g3_descriptor": "concentration_weighted_rms_radius_mismatch",
        "shared_initial_batch_per_prototype": POP_PER_PROTO,
        "evaluations_per_prototype_per_arm": EVALUATIONS_PER_PROTO,
        "sha256": {name: sha256(paths[name]) for name in required},
    }
    temporary = paths["checkpoint"].with_name(paths["checkpoint"].name + ".staging")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, paths["checkpoint"])


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    margin = z * math.sqrt(proportion * (1.0 - proportion) / total + z**2 / (4.0 * total**2)) / denominator
    return float(center - margin), float(center + margin)


def calibrate_control_thresholds(
    known: pd.DataFrame,
    *,
    bootstrap_repetitions: int = 20000,
) -> tuple[dict[str, object], pd.DataFrame]:
    reached = known.loc[known["pass_g1_g2"]].copy().reset_index(drop=True)
    values = reached["radius_mismatch"].to_numpy(float)
    if len(values) < 3 or not np.isfinite(values).all():
        raise RuntimeError("Insufficient positive controls for G3 calibration")
    rng = np.random.default_rng(42003)
    rows: list[dict[str, object]] = []
    policies: dict[str, object] = {}
    for label, quantile in [("P90", 0.90), ("P95", 0.95)]:
        threshold = float(np.quantile(values, quantile))
        loo_thresholds = np.array(
            [float(np.quantile(np.delete(values, index), quantile)) for index in range(len(values))]
        )
        loo_pass = values <= loo_thresholds
        successes = int(loo_pass.sum())
        wilson_low, wilson_high = wilson_interval(successes, len(values))
        bootstrap = np.empty(bootstrap_repetitions, dtype=float)
        for iteration in range(bootstrap_repetitions):
            sample = rng.choice(values, size=len(values), replace=True)
            bootstrap[iteration] = np.quantile(sample, quantile)
            rows.append(
                {
                    "iteration": iteration + 1,
                    "policy": label,
                    "threshold": float(bootstrap[iteration]),
                }
            )
        policies[label] = {
            "quantile": quantile,
            "full_control_threshold": threshold,
            "leave_one_control_out_successes": successes,
            "leave_one_control_out_total": int(len(values)),
            "leave_one_control_out_retention": float(successes / len(values)),
            "leave_one_control_out_wilson_ci95": [wilson_low, wilson_high],
            "threshold_bootstrap_ci95": [
                float(np.percentile(bootstrap, 2.5)),
                float(np.percentile(bootstrap, 97.5)),
            ],
            "threshold_bootstrap_repetitions": bootstrap_repetitions,
        }
    return {
        "g1_g2_denominator": int(len(values)),
        "descriptor": "concentration-weighted RMS relative radius mismatch",
        "policies": policies,
    }, pd.DataFrame(rows)


def main() -> None:
    start = time.time()
    evaluator = CandidateEvaluator()
    model_sha256 = sha256(MODELS_DIR / "catboost_full_context.joblib")
    runtime = runtime_fingerprint()
    all_summaries = []
    all_histories = []
    threshold_rows = []

    # Validate the fast Pareto implementation against a brute-force small case.
    rng_check = np.random.default_rng(7)
    check_a, check_b = rng_check.random(40), rng_check.random(40)
    fast = pareto_ranks(check_a, check_b)
    brute = np.zeros(40, dtype=int)
    remaining = set(range(40))
    rank = 0
    while remaining:
        front = []
        for i in remaining:
            dominated = any(
                (check_a[j] >= check_a[i] and check_b[j] >= check_b[i])
                and (check_a[j] > check_a[i] or check_b[j] > check_b[i])
                for j in remaining
                if j != i
            )
            if not dominated:
                front.append(i)
        for i in front:
            brute[i] = rank
            remaining.remove(i)
        rank += 1
    if not np.array_equal(fast, brute):
        raise AssertionError("Fast Pareto rank implementation failed self-check")

    # Calibrate G3 before the candidate benchmark.  P95 is the reference
    # inclusion policy; P90 defines a nested, more conservative core.
    assignment, known_initial = build_known_controls(evaluator)
    control_policy, threshold_bootstrap = calibrate_control_thresholds(known_initial)
    control_p90 = float(control_policy["policies"]["P90"]["full_control_threshold"])
    control_p95 = float(control_policy["policies"]["P95"]["full_control_threshold"])
    known = screen(known_initial, radius_threshold=control_p95)
    known["g3_p90_pass"] = known["pass_g1_g2"] & known["radius_mismatch"].le(control_p90)
    known["g3_p95_pass"] = known["pass_g1_g2"] & known["radius_mismatch"].le(control_p95)
    atomic_csv(threshold_bootstrap, RESULTS_DIR / "control_threshold_bootstrap.csv")

    for seed in SEEDS:
        paths = seed_artifacts(seed)
        if checkpoint_is_valid(seed, model_sha256, str(runtime["sha256"])):
            log(f"Loading validated checkpoint for seed {seed}")
            all_summaries.extend(read_checkpoint_csv(paths["summary"]).to_dict(orient="records"))
            all_histories.append(read_checkpoint_csv(paths["history"]))
            threshold_rows.extend(read_checkpoint_csv(paths["threshold"]).to_dict(orient="records"))
            continue

        log(f"Starting matched search benchmark seed {seed}")
        initial = shared_initial_population(seed)
        nsga_pool, history = nsga_proposal(seed, evaluator, initial)
        random_pool = random_proposal(seed, evaluator, initial)
        nsga_initial = nsga_pool.loc[nsga_pool["initial_shared_batch"], ["proto_name", "canonical_key"]].reset_index(drop=True)
        random_initial = random_pool.loc[random_pool["initial_shared_batch"], ["proto_name", "canonical_key"]].reset_index(drop=True)
        if len(nsga_initial) != POP_PER_PROTO * len(prototypes()) or not nsga_initial.equals(random_initial):
            raise AssertionError("Matched arms do not share an identical first proposal batch")

        seed_summaries: list[dict[str, object]] = []
        seed_thresholds: list[dict[str, object]] = []
        for proposal, pool in [("nsga2", nsga_pool), ("random", random_pool)]:
            # Preserve one row per prototype/composition after equal objective budgets.
            unique_pool = (
                pool.sort_values(
                    ["generation", "y_discounted", "canonical_key", "proto_name"],
                    ascending=[True, False, True, True],
                    kind="mergesort",
                )
                .drop_duplicates(["proto_name", "canonical_key"])
                .reset_index(drop=True)
            )
            pool_path = paths["nsga_pool"] if proposal == "nsga2" else paths["random_pool"]
            atomic_csv(unique_pool, pool_path, gzip_output=True)
            # Select from the persisted representation so fresh and resumed runs agree bit-for-bit.
            unique_pool = read_checkpoint_csv(pool_path)

            for floor_on in [True, False]:
                topk = select_topk(unique_pool, floor_on=floor_on)
                screened = screen(topk, radius_threshold=control_p95)
                screened["g3_p90_pass"] = screened["pass_g1_g2"] & screened["radius_mismatch"].le(control_p90)
                screened["g3_p95_pass"] = screened["pass_g1_g2"] & screened["radius_mismatch"].le(control_p95)
                arm = f"{proposal}__floor_{'on' if floor_on else 'off'}"
                screened["seed"] = seed
                screened["arm"] = arm
                atomic_csv(screened, SEARCH_DIR / f"{arm}_seed{seed}_topk_screened.csv")
                seed_summaries.append(arm_summary(topk, screened, seed, proposal, floor_on))

                for threshold in np.round(np.arange(0.05, 0.301, 0.01), 2):
                    reached = screened["pass_g1_g2"]
                    retained = reached & screened["radius_mismatch"].le(float(threshold))
                    seed_thresholds.append(
                        {
                            "pool": arm,
                            "seed": seed,
                            "radius_threshold": float(threshold),
                            "n_reaching_g3": int(reached.sum()),
                            "n_passing_g3": int(retained.sum()),
                            "conditional_retention": float(retained.sum() / reached.sum()) if reached.sum() else np.nan,
                        }
                    )

        seed_summary_frame = pd.DataFrame(seed_summaries)
        seed_threshold_frame = pd.DataFrame(seed_thresholds)
        atomic_csv(history, paths["history"])
        atomic_csv(seed_summary_frame, paths["summary"])
        atomic_csv(seed_threshold_frame, paths["threshold"])
        write_seed_checkpoint(seed, model_sha256, str(runtime["sha256"]))
        all_summaries.extend(seed_summaries)
        all_histories.append(history)
        threshold_rows.extend(seed_thresholds)
        log(f"Completed seed {seed}")

    summary = pd.DataFrame(all_summaries)
    atomic_csv(summary, RESULTS_DIR / "search_seed_summary.csv")
    atomic_csv(pd.concat(all_histories, ignore_index=True), RESULTS_DIR / "nsga_generation_history.csv")
    atomic_csv(pd.DataFrame(threshold_rows), RESULTS_DIR / "radius_threshold_sensitivity.csv")
    paired = summarize_seed_pairs(summary)
    atomic_csv(paired, RESULTS_DIR / "search_paired_seed_comparisons.csv")

    atomic_csv(assignment, RESULTS_DIR / "known_control_prototype_assignment.csv")
    atomic_csv(known, RESULTS_DIR / "known_control_screened.csv")
    known_reached = known.loc[known["pass_g1_g2"]].copy()
    known_threshold = []
    for threshold in np.round(np.arange(0.05, 0.301, 0.01), 2):
        passed = known_reached["radius_mismatch"].le(float(threshold))
        known_threshold.append(
            {
                "pool": "known_positive_controls",
                "seed": -1,
                "radius_threshold": float(threshold),
                "n_reaching_g3": int(len(known_reached)),
                "n_passing_g3": int(passed.sum()),
                "conditional_retention": float(passed.mean()) if len(passed) else np.nan,
            }
        )
    threshold_frame = read_checkpoint_csv(RESULTS_DIR / "radius_threshold_sensitivity.csv")
    threshold_frame = pd.concat([threshold_frame, pd.DataFrame(known_threshold)], ignore_index=True)
    atomic_csv(threshold_frame, RESULTS_DIR / "radius_threshold_sensitivity.csv")

    known_summary = {
        "canonical_compositions_total": int(len(assignment)),
        "template_applicable_controls": int(len(known)),
        "template_not_applicable_controls": int(assignment["proto_name"].isna().sum()),
        "g1_survivors": int(known["pass_g1"].sum()),
        "g2_survivors": int(known["pass_g1_g2"].sum()),
        "g3_conservative_p90_survivors": int(known["g3_p90_pass"].sum()),
        "g3_reference_p95_survivors": int(known["g3_p95_pass"].sum()),
        "g4_survivors": int(known["pass_g1_g2_g3_g4"].sum()),
        "positive_control_radius_p90_after_g1_g2": control_p90,
        "positive_control_radius_p95_after_g1_g2": control_p95,
        "cross_fitted_policy_validation": control_policy,
        "specificity_estimable": False,
    }
    (RESULTS_DIR / "known_control_summary.json").write_text(json.dumps(known_summary, indent=2), encoding="utf-8")

    floor_on_effect = paired[
        (paired["comparison"] == "proposal_effect_floor_on")
        & (paired["metric"] == "feasible_hypervolume")
    ].iloc[0]
    keep_nsga_as_performance_claim = bool(
        floor_on_effect["mean_paired_delta"] > 0
        and floor_on_effect["paired_bootstrap_ci95_low"] > 0
    )
    decision = {
        "primary_metric": "post_screen_feasible_hypervolume",
        "proposal_effect_floor_on_mean_paired_delta": float(floor_on_effect["mean_paired_delta"]),
        "proposal_effect_floor_on_positive_seeds": int(floor_on_effect["n_positive"]),
        "proposal_effect_floor_on_total_seeds": int(floor_on_effect["n_seeds"]),
        "proposal_effect_floor_on_paired_bootstrap_ci95": [
            float(floor_on_effect["paired_bootstrap_ci95_low"]),
            float(floor_on_effect["paired_bootstrap_ci95_high"]),
        ],
        "retain_nsga2_as_substantive_performance_claim": keep_nsga_as_performance_claim,
        "interpretation": (
            "Matched seeds support a consistent feasible-hypervolume gain."
            if keep_nsga_as_performance_claim
            else "Matched seeds do not support a robust NSGA-II necessity claim; treat it as a replaceable search implementation."
        ),
    }
    (RESULTS_DIR / "nsga_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    results_path = RESULTS_DIR / "results_for_manuscript.json"
    manuscript = json.loads(results_path.read_text(encoding="utf-8"))
    summary_agg = summary.groupby("arm").agg(
        {
            "y_pred_mean": ["mean", "std"],
            "y_discounted_mean": ["mean", "std"],
            "model_distance_mean": ["mean", "std"],
            "comp_novelty_mean": ["mean", "std"],
            "prototype_coverage": ["mean", "std"],
            "g4_survivors": ["mean", "std"],
            "feasible_hypervolume": ["mean", "std"],
        }
    )
    summary_agg.columns = ["_".join(parts) for parts in summary_agg.columns]
    manuscript["search_benchmark"] = {
        "seeds": SEEDS,
        "evaluations_per_prototype": EVALUATIONS_PER_PROTO,
        "summary_mean_sd": summary_agg.to_dict(orient="index"),
        "decision": decision,
    }
    manuscript["screening"] = {
        "primary_radius_policy": {"name": "positive-control P95", "threshold": control_p95},
        "nested_conservative_radius_policy": {"name": "positive-control P90", "threshold": control_p90},
        "known_controls": known_summary,
        "g3_role": "concentration-weighted RMS risk descriptor with a cross-fitted P95 reference policy and nested P90 conservative core",
        "jt_role": "supplementary proxy, not a hard core gate",
    }
    results_path.write_text(json.dumps(manuscript, indent=2), encoding="utf-8")

    config = {
        "seeds": SEEDS,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "algorithm": "canonical elitist NSGA-II with parent-offspring union",
        "population_per_prototype": POP_PER_PROTO,
        "generations": N_GENERATIONS,
        "evaluations_per_prototype": EVALUATIONS_PER_PROTO,
        "topk": TOPK,
        "floor_per_prototype": FLOOR_PER_PROTO,
        "fixed_context": {
            "temperature_K": FIXED_TEMPERATURE_K,
            "current_rate_C": FIXED_CURRENT_RATE_C,
            "cycle": FIXED_CYCLE,
        },
        "distance_discount": {"k": KNN_K, "reference": "training_unique_composition_p95", "decay": DISTANCE_DECAY},
        "hard_knn_gate": False,
        "chemistry_filter_inside_search": False,
        "shared_initial_batch_per_prototype_and_seed": POP_PER_PROTO,
        "ranking_shared_across_arms": "global final Pareto rank + crowding distance + discounted capacity",
        "determinism": {
            "catboost_prediction_threads": 1,
            "nearest_neighbor_threads": 1,
            "csv_float_format": "%.17g",
            "gzip_mtime": 0,
            "model_sha256": model_sha256,
            "runtime_fingerprint": runtime,
        },
        "parameter_space_projection": {
            "applied_to": "every NSGA-II and random proposal before evaluation",
            "li_site_total": [min(LI_EXCESS_RANGE), max(LI_EXCESS_RANGE)],
            "maximum_nonbase_framework_dopant_fraction": MAX_DOPANT_FRAC,
            "framework_cation_total_preserved": True,
            "anion_total_preserved": True,
            "runtime_assertion": "parameter_bounds_ok for every evaluated candidate",
        },
        "screening": {
            "G1": "charge-neutrality plus Pauling ordering",
            "G2": "prototype Li/TM window",
            "G3": {
                "descriptor": "concentration-weighted RMS relative radius mismatch",
                "legacy_maximum_mismatch": "diagnostic only",
                "primary_policy": {"name": "positive-control P95", "threshold": control_p95},
                "nested_conservative_policy": {"name": "positive-control P90", "threshold": control_p90},
                "validation": "leave-one-control-out retention plus 20,000-bootstrap threshold intervals",
            },
            "G4": "predicted capacity <= conservative cationic redox budget",
            "JT": "supplementary proxy only",
        },
    }
    (CONFIG_DIR / "search_screen_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    log(f"Search/screen benchmark complete in {(time.time() - start) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
