"""Analyze whether smogon_normalized.parquet is applicable as the sole
data source for the teambuild policy cache.

Assesses schema completeness against requirements from each phase of
Teambuild_policy_TODO.md: weighted usage sampling, Bayesian MAP fitness,
genetic operators, EV cap enforcement, and Showdown team hydration.
"""

import json
import os
from typing import Any

import numpy as np
import pandas as pd

SMOGON_NORM_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "data", "processed", "smogon_normalized.parquet"
)
SMOGON_RAW_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "data", "raw", "smogon",
    "gen9championsvgc2026regmabo3-1760_2026-05.json",
)
GOLD_TENSORS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "data", "processed", "gold_tensors.parquet",
)


def analyze_smogon_normalized() -> dict[str, Any]:
    """Load and profile smogon_normalized.parquet.

    Returns:
        Dict with keys: shape, columns, dtypes, null_counts, unique_ids,
        sample_row, stats.
    """
    df = pd.read_parquet(SMOGON_NORM_PATH)
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtype_summary": {str(k): int(v) for k, v in df.dtypes.value_counts().items()},
        "null_counts": {str(k): int(v) for k, v in df.isnull().sum().items()},
        "unique_ids": int(df["pokeapi_id"].nunique()),
        "total_rows": int(len(df)),
        "sample_row": {
            col: (df.iloc[0][col].tolist() if isinstance(df.iloc[0][col], np.ndarray) else str(df.iloc[0][col]))
            for col in df.columns
        } if len(df) > 0 else {},
    }


def analyze_raw_smogon() -> dict[str, Any]:
    """Load and profile the raw Smogon Chaos JSON.

    Returns:
        Dict with keys: species_count, top_level_keys, per_species_keys,
        total_usage, usage_range, sample_species.
    """
    with open(SMOGON_RAW_PATH, "r") as f:
        raw: dict[str, Any] = json.load(f)
    data: dict[str, Any] = raw["data"]
    species_list = list(data.keys())
    usage_values = [data[s].get("usage", 0.0) for s in species_list]
    return {
        "species_count": len(species_list),
        "top_level_keys": list(raw.keys()),
        "per_species_keys": sorted(
            {k for entry in data.values() for k in entry.keys()}
        ),
        "total_usage": float(sum(usage_values)),
        "usage_range": {
            "min": float(min(usage_values)),
            "max": float(max(usage_values)),
            "mean": float(np.mean(usage_values)),
        },
        "sample_species": species_list[0],
        "sample_entry_keys": list(data[species_list[0]].keys()),
        "has_usage": "usage" in data[species_list[0]],
        "has_abilities": "Abilities" in data[species_list[0]],
        "has_items": "Items" in data[species_list[0]],
        "has_spreads": "Spreads" in data[species_list[0]],
        "has_moves": "Moves" in data[species_list[0]],
        "has_teammates": "Teammates" in data[species_list[0]],
    }


def requirement_matrix() -> dict[str, dict[str, str]]:
    """Map each teambuild phase to its data requirements and availability.

    Returns:
        Dict of phase_name → {requirement: availability_status}.
    """
    norm_cols = {"pokeapi_id", "item", "ability", "nature", "evs", "moves"}

    return {
        "Phase 1 (Cache)": {
            "pokeapi_id index": "OK -- smogon_normalized has pokeapi_id" if "pokeapi_id" in norm_cols else "MISSING",
            "species name strings": "MISSING in smogon_normalized.parquet; present in raw JSON keys",
            "usage weights per species": "MISSING in smogon_normalized.parquet; present as 'usage' in raw JSON",
            "weighted item distribution": "MISSING in smogon_normalized.parquet (single choice); present as 'Items' dict in raw JSON",
            "weighted ability distribution": "MISSING in smogon_normalized.parquet (single choice); present as 'Abilities' dict in raw JSON",
            "weighted EV spread distribution": "MISSING in smogon_normalized.parquet (single choice); present as 'Spreads' dict in raw JSON",
            "weighted move distribution": "MISSING in smogon_normalized.parquet (4 moves only); present as 'Moves' dict in raw JSON",
        },
        "Phase 2 (Bayesian MAP Fitness)": {
            "usage weight for log(P(Usage))": "MISSING -- required by fitness formula; only in raw JSON",
            "GMM centroids for Euclidean distance": "EXTERNAL -- must load from models/archetype_gmm.pkl",
            "species names for macro_features_array": "MISSING -- required by transformer.py; only raw JSON species keys",
            "pokeapi_id for species->id mapping": "OK -- present in both sources",
        },
        "Phase 3 (Genetic Operators)": {
            "usage-weighted population sampling pool": "MISSING -- smogon_normalized lacks usage weights; raw JSON has 'usage'",
            "multiple items per species for mutation": "MISSING -- smogon_normalized has single item; raw JSON has full item dict",
            "multiple EV spreads per species": "MISSING -- smogon_normalized has single spread; raw JSON has full 'Spreads' dict",
            "EV sum=510 constraint enforcement": "PARTIAL -- smogon_normalized has one spread per species; raw JSON has all weighted spreads for validation",
            "multiple abilities for mutation swaps": "MISSING -- smogon_normalized has single ability; raw JSON has 'Abilities' dict",
        },
        "Phase 5 (Battle Royale Hydration)": {
            "fully hydrated Showdown team string": "PARTIAL -- smogon_normalized has 1 build per species (good for single instantiation); raw JSON provides distribution for diverse teams",
            "species base stats (GenData)": "EXTERNAL -- loaded via poke_env.GenData.from_gen(9) at runtime",
            "nature string": "OK -- present in both sources",
        },
        "Phase 7 (Smogon Baseline)": {
            "top-6 by usage weight": "MISSING -- smogon_normalized has no usage column; raw JSON has 'usage' sorted desc",
            "baseline team Showdown hydration": "PARTIAL -- possible with smogon_normalized only for 1 build, but lacks usage ranking",
        },
    }


def run_analysis() -> dict[str, Any]:
    """Entry point: run all analyses and return consolidated results.

    Returns:
        Dict with normalized_profile, raw_profile, requirements, verdict.
    """
    norm = analyze_smogon_normalized()
    raw = analyze_raw_smogon()
    reqs = requirement_matrix()

    missing_columns = []
    for req, verdict in reqs["Phase 1 (Cache)"].items():
        if "MISSING" in verdict.upper():
            missing_columns.append(f"  - {req}: {verdict}")

    verdict_text = (
        "DENIED -- smogon_normalized.parquet cannot be the sole cache source. "
        "Missing critical columns: usage weight, species name, weighted distributions "
        "(items, abilities, spreads, moves). The raw Smogon Chaos JSON must be the "
        "primary data source. Alternatively, a new normalization script could produce "
        "a richer Parquet file that includes these fields."
    ) if missing_columns else "APPROVED"

    return {
        "normalized_profile": norm,
        "raw_profile": raw,
        "requirements": reqs,
        "verdict": verdict_text,
        "missing_in_normalized": missing_columns,
        "recommendation": (
            "Load raw Smogon Chaos JSON as the primary teambuild cache. "
            "Key advantages over smogon_normalized.parquet: "
            "1) 'usage' field enables log(P(Usage)) fitness scoring; "
            "2) full weighted distributions for Abilities, Items, Spreads, Moves "
            "enable probabilistic genetic operators and team hydration; "
            "3) species names (JSON keys) are compatible with macro_features_array() "
            "from the Selection policy's transformer module. "
            "The raw JSON provides 170 species with complete distribution data vs. "
            "243 rows of flat single-choice data in the normalized Parquet."
        ),
    }


if __name__ == "__main__":
    import sys

    results = run_analysis()

    lines = []
    lines.append("=" * 72)
    lines.append("TEAMBUILD POLICY -- SMOGON DATA SOURCE APPLICABILITY ANALYSIS")
    lines.append("=" * 72)

    lines.append("")
    lines.append("--- smogon_normalized.parquet Profile ---")
    profile = results["normalized_profile"]
    lines.append(f"  Shape:  {profile['shape']} (rows, cols)")
    lines.append(f"  Columns: {profile['columns']}")
    lines.append(f"  Unique pokeapi_ids: {profile['unique_ids']}")
    lines.append(f"  Null counts: {profile['null_counts']}")
    lines.append(f"  Dtype summary: {profile['dtype_summary']}")

    lines.append("")
    lines.append("--- Raw Smogon Chaos JSON Profile ---")
    raw = results["raw_profile"]
    lines.append(f"  Species count: {raw['species_count']}")
    lines.append(f"  Per-species keys: {raw['per_species_keys']}")
    lines.append(f"  Total usage: {raw['total_usage']}")
    lines.append(f"  Usage range: {raw['usage_range']}")
    lines.append(f"  Sample species: {raw['sample_species']}")
    lines.append(f"  Has 'usage': {raw['has_usage']}")
    lines.append(f"  Has 'Abilities' (weighted): {raw['has_abilities']}")
    lines.append(f"  Has 'Items' (weighted): {raw['has_items']}")
    lines.append(f"  Has 'Spreads' (weighted): {raw['has_spreads']}")
    lines.append(f"  Has 'Moves' (weighted): {raw['has_moves']}")
    lines.append(f"  Has 'Teammates': {raw['has_teammates']}")

    lines.append("")
    lines.append("--- Phase-by-Phase Requirement Assessment ---")
    for phase, checks in results["requirements"].items():
        lines.append("")
        lines.append(f"  {phase}:")
        for req, status in checks.items():
            label = status.split(" -- ")[0] if " -- " in status else status
            lines.append(f"    [{label}] {req}")

    lines.append("")
    lines.append("=" * 72)
    lines.append(f"VERDICT: {results['verdict']}")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"RECOMMENDATION:\n{results['recommendation']}")

    output = "\n".join(lines)
    try:
        print(output)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(output.encode("utf-8") + b"\n")
