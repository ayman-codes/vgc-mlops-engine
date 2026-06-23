"""In-memory cache loading raw Smogon Chaos JSON as the teambuild data source.

Exposes species_keys, usage_weights, the full distribution data, and helper
methods for weighted random sampling, Pokemon hydration, and Showdown
serialization.
"""

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.agent.base import Pokemon, Move
from src.agent.selection_policy.double_oracle import Strategy, _pokemon_to_showdown

SMOGON_JSON_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "data", "raw", "smogon",
    "gen9championsvgc2026regmabo3-1760_2026-05.json",
)
GMM_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "models", "archetype_gmm.pkl",
)


def _parse_spread(spread_key: str) -> tuple[str, list[int]]:
    """Parse a Smogon Chaos spread key into nature and EV vector.

    Spread format: "Nature:HP/Atk/Def/SpA/SpD/Spe"
    where values are Chaos-encoded integers (0-63 range).

    Args:
        spread_key: Spread string from the Chaos JSON Spreads dict.

    Returns:
        Tuple of (nature string, [hp, atk, def, spa, spd, spe] int list).
    """
    parts = spread_key.split(":")
    nature = parts[0]
    ev_string = parts[1] if len(parts) > 1 else "0/0/0/0/0/0"
    evs = [int(x) for x in ev_string.split("/")]
    return nature, evs


def _weighted_choice(options: dict[str, float]) -> str:
    """Pick a key from a weighted dict by probability proportional to weight.

    Args:
        options: Mapping of key to numeric weight.

    Returns:
        Selected key string, or empty string if options is empty.
    """
    if not options:
        return ""
    items = list(options.keys())
    weights = np.array([options[k] for k in items], dtype=np.float64)
    total = float(weights.sum())
    if total <= 0:
        return random.choice(items)
    probs = weights / total
    r = random.random()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if r <= cumulative:
            return items[i]
    return items[-1]


def _top_weighted_moves(moves_pool: dict[str, float], n: int) -> list[str]:
    """Select the top-n moves from a weighted pool by descending weight.

    Args:
        moves_pool: Mapping of move name to usage weight.
        n: Number of moves to select.

    Returns:
        List of up to n move names sorted by weight descending.
    """
    if not moves_pool:
        return []
    items = list(moves_pool.keys())
    if n >= len(items):
        return items[:n]
    sorted_items = sorted(items, key=lambda k: moves_pool[k], reverse=True)
    return sorted_items[:n]


@dataclass
class TeambuildCache:
    """In-memory cache aggregating Smogon distribution data and GMM model.

    Loads the raw Smogon Chaos JSON, pre-trained GMM archetype model,
    and exposes helper methods for weighted sampling from distribution
    data, Pokemon hydration, and Showdown team serialization.

    Args:
        species_keys: List of 170 lowercase species names (parallel arrays
            with usage_weights).
        usage_weights: Normalized usage weights summing to 1.0, shape (170,).
        raw_data: Full JSON data dict keyed by Showdown species name.
        gmm: Fitted GaussianMixture from models/archetype_gmm.pkl.
        scaler: StandardScaler from models/archetype_gmm.pkl, or None.
    """

    species_keys: list[str] = field(default_factory=list)
    usage_weights: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    raw_data: dict[str, dict[str, Any]] = field(default_factory=dict)
    gmm: GaussianMixture | None = None
    scaler: StandardScaler | None = None

    @property
    def n_species(self) -> int:
        """Number of species available in the cache."""
        return len(self.species_keys)

    def get_entry(self, species: str) -> dict[str, Any] | None:
        """Look up a raw JSON entry by species name.

        Performs a case-insensitive search against the raw data keys.

        Args:
            species: Species name (any case), e.g. "basculegion" or "FlutterMane".

        Returns:
            Full Chaos JSON entry dict, or None if unknown.
        """
        species_lower = species.lower()
        for key, entry in self.raw_data.items():
            if key.lower() == species_lower:
                return entry
        return None

    def sample_ability(self, species: str) -> str:
        """Select an ability via weighted random sampling from species data.

        Args:
            species: Lowercase species name.

        Returns:
            Ability string (lowercase), or empty string if no data.
        """
        entry = self.get_entry(species)
        if entry is None:
            return ""
        abilities = entry.get("Abilities", {})
        return _weighted_choice(abilities)

    def sample_item(self, species: str) -> str:
        """Select a held item via weighted random sampling from species data.

        Args:
            species: Lowercase species name.

        Returns:
            Item string (lowercase), or empty string if no data.
        """
        entry = self.get_entry(species)
        if entry is None:
            return ""
        items = entry.get("Items", {})
        return _weighted_choice(items)

    def sample_spread(self, species: str) -> tuple[str, list[int]]:
        """Select an EV spread via weighted random sampling from species data.

        Args:
            species: Lowercase species name.

        Returns:
            Tuple of (nature string, [hp, atk, def, spa, spd, spe] int list).
        """
        entry = self.get_entry(species)
        if entry is None:
            return ("serious", [0, 0, 0, 0, 0, 0])
        spreads = entry.get("Spreads", {})
        spread_key = _weighted_choice(spreads)
        if not spread_key:
            return ("serious", [0, 0, 0, 0, 0, 0])
        return _parse_spread(spread_key)

    def sample_moves(self, species: str, n: int = 4) -> list[str]:
        """Select the top-n moves by usage weight from species data.

        Args:
            species: Lowercase species name.
            n: Maximum number of moves to select.

        Returns:
            List of up to n move names (lowercase), ordered by weight.
        """
        entry = self.get_entry(species)
        if entry is None:
            return []
        moves_pool = entry.get("Moves", {})
        return _top_weighted_moves(moves_pool, n)

    def hydrate_pokemon(self, species: str) -> Pokemon:
        """Build a fully hydrated Pokemon from weighted distribution data.

        Samples ability, item, spread, and moves from the Smogon Chaos
        usage distributions for the given species. The species name is
        normalized to lowercase for consistency with the Pokemon dataclass
        and downstream macro-feature extraction.

        Falls back to a minimal Pokemon instance when species data is
        unavailable.

        Args:
            species: Species name (any case).

        Returns:
            A fully populated Pokemon instance.
        """
        species_lower = species.lower()
        entry = self.get_entry(species_lower)
        if entry is None:
            return Pokemon(species=species_lower)

        ability = self.sample_ability(species_lower)
        item = self.sample_item(species_lower)
        nature, evs = self.sample_spread(species_lower)
        moves = self.sample_moves(species_lower)

        return Pokemon(
            species=species_lower,
            ability=ability,
            item=item,
            nature=nature,
            ev_hp=evs[0],
            ev_atk=evs[1],
            ev_def=evs[2],
            ev_spa=evs[3],
            ev_spd=evs[4],
            ev_spe=evs[5] if len(evs) > 5 else 0,
            moves=[Move(name=m) for m in moves],
        )

    def hydrate_team(self, indices: list[int]) -> list[Pokemon]:
        """Build 6 fully hydrated Pokemon from species index array.

        Args:
            indices: List of exactly 6 integer indices into self.species_keys.

        Returns:
            List of 6 hydrated Pokemon instances.
        """
        return [self.hydrate_pokemon(self.species_keys[i]) for i in indices]

    def team_to_showdown(self, team: list[Pokemon], strategy: Strategy) -> str:
        """Convert a hydrated team and ordering strategy to a Showdown team string.

        Reuses _pokemon_to_showdown from the Selection policy's double_oracle
        module for consistent serialization format.

        Args:
            team: List of 6 hydrated Pokemon.
            strategy: 6-element index tuple defining the team order
                (lead1, lead2, back1, back2, back3, back4).

        Returns:
            Showdown-format team string with Pokemon separated by double newlines.
        """
        return "\n\n".join(
            _pokemon_to_showdown(team[i]) for i in strategy
        )


def load_teambuild_cache(
    smogon_path: str = SMOGON_JSON_PATH,
    gmm_path: str = GMM_MODEL_PATH,
) -> TeambuildCache:
    """Load the Smogon Chaos JSON and GMM model into an in-memory cache.

    Builds parallel species_keys and usage_weights arrays indexed 0-169
    for fast genetic algorithm operations.

    Args:
        smogon_path: Path to the Smogon Chaos JSON file.
        gmm_path: Path to the serialized GMM model pickle.

    Returns:
        A fully initialized TeambuildCache instance.

    Raises:
        FileNotFoundError: If either required file does not exist.
    """
    if not os.path.exists(smogon_path):
        raise FileNotFoundError(
            f"Smogon Chaos JSON not found: {smogon_path}"
        )
    if not os.path.exists(gmm_path):
        raise FileNotFoundError(
            f"GMM model not found: {gmm_path}"
        )

    with open(smogon_path, "r") as f:
        raw: dict[str, Any] = json.load(f)
    data: dict[str, dict[str, Any]] = raw["data"]

    sorted_raw_keys = sorted(
        data.keys(),
        key=lambda k: data[k].get("usage", 0),
        reverse=True,
    )
    species_keys = [k.lower() for k in sorted_raw_keys]
    usage_vals = np.array(
        [data[k].get("usage", 0.0) for k in sorted_raw_keys],
        dtype=np.float64,
    )
    usage_weights = usage_vals / usage_vals.sum()

    gmm_loaded: Any = joblib.load(gmm_path)
    if isinstance(gmm_loaded, dict):
        gmm = gmm_loaded["gmm"]
        scaler = gmm_loaded.get("scaler", None)
    else:
        gmm = gmm_loaded
        scaler = None

    return TeambuildCache(
        species_keys=species_keys,
        usage_weights=usage_weights,
        raw_data=data,
        gmm=gmm,
        scaler=scaler,
    )
