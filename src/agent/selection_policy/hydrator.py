"""Real-data Pokémon team hydrator from Smogon usage statistics.

Loads Gen 9 VGC usage JSON and constructs Pokemon objects with
variance-weighted random selection over items, abilities, spreads, and moves.
Supports archetype-aware counter-building when a distribution is provided.
"""

import json
import os
import random
from typing import Any

from src.agent.base import Pokemon, Move

_SP_DEF_ITEMS: set[str] = {
    "assaultvest", "shucaberry", "passhoberry", "wacanberry",
    "roseliberry", "habanberry", "yacheberry", "payapaberry",
    "rindoberry", "cobaberry",
}

_PHYS_DEF_ITEMS: set[str] = {
    "rockyhelmet", "chopleberry", "colburberry", "babiriberry",
    "kebiaberry", "tangaberry", "chartiberry", "occaberry",
    "kasibberry", "chestoberry", "chilanberry",
}

SMOGON_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "raw", "smogon", "gen9championsvgc2026regmabo3-1760_2026-05.json")

_smogmon_data: dict[str, dict[str, Any]] | None = None


def _load_smogmon() -> dict[str, dict[str, Any]]:
    """Lazy-load the Smogon usage JSON into a lowercased species→entry dict.

    Returns:
        Dict mapping lowercase species names to their full Smogon entries.
    """
    global _smogmon_data
    if _smogmon_data is None:
        with open(SMOGON_PATH) as f:
            raw: dict[str, Any] = json.load(f)
        raw_data: dict[str, Any] = raw["data"]
        _smogmon_data = {k.lower(): v for k, v in raw_data.items()}
    return _smogmon_data


def _weighted_choice(
    options: dict[str, float],
    variance: float = 0.20,
    counter_mult: dict[str, float] | None = None,
) -> str:
    """Pick a key from a weighted dict with optional noise and counter-bias.

    Args:
        options: Mapping of key → base weight (e.g. usage percentage).
        variance: Noise level (0 = deterministic, 1 = uniform random).
        counter_mult: Optional per-key multiplier to bias selection toward
            or away from certain items/spreads.

    Returns:
        Selected key string, or empty string if options is empty.
    """
    if not options:
        return ""
    items = list(options.keys())
    weights = [options[k] * counter_mult.get(k, 1.0) for k in items] if counter_mult else [options[k] for k in items]
    total = sum(weights)
    if total <= 0:
        return random.choice(items)
    probs = [w / total for w in weights]
    adjusted = [p * (1.0 - variance) + (random.random() * variance) for p in probs]
    adjusted_total = sum(adjusted)
    if adjusted_total <= 0:
        return random.choice(items)
    adjusted_norm = [a / adjusted_total for a in adjusted]
    r = random.random()
    cumulative = 0.0
    for i, p in enumerate(adjusted_norm):
        cumulative += p
        if r <= cumulative:
            return items[i]
    return items[-1]


def _parse_spread(spread_key: str) -> tuple[str, list[int]]:
    """Parse a Smogon spread key into nature and EV vector.

    Smogon format: "Nature:HP/Atk/Def/SpA/SpD/Spe".

    Args:
        spread_key: Spread string from Smogon data.

    Returns:
        Tuple of (nature string, [hp, atk, def, spa, spd, spe] int list).
    """
    parts = spread_key.split(":")
    nature = parts[0]
    ev_string = parts[1] if len(parts) > 1 else "0/0/0/0/0/0"
    evs = [int(x) for x in ev_string.split("/")]
    return nature, evs


def _filter_valid_spreads(spreads: dict[str, float]) -> dict[str, float]:
    """Remove spread entries whose parsed EV sum is zero.

    Zero-EV spreads originate from players who forgot to set EVs on
    Pokemon Showdown; Smogon still logs them. Allowing these into the
    hydrator would push disadvantaged 0/0/0/0/0/0 Pokemon into live
    poke-env battles.

    Args:
        spreads: Mapping of spread key to usage weight.

    Returns:
        Filtered dict containing only spreads with positive EV investment.
    """
    return {
        k: v for k, v in spreads.items()
        if sum(_parse_spread(k)[1]) > 0
    }


def _get_dominant_archetype(archetype_distribution: list[float] | None) -> int | None:
    """Return the index of the most probable archetype cluster.

    Args:
        archetype_distribution: Probabilities per cluster, or None.

    Returns:
        Cluster index with highest probability, or None if input is None/empty.
    """
    if not archetype_distribution:
        return None
    return max(range(len(archetype_distribution)), key=lambda i: archetype_distribution[i])


def _item_counter_multiplier(item: str, archetype_idx: int) -> float:
    """Return a weight multiplier for an item given the target archetype.

    Specially-defensive items are favoured against fast-special (idx 0);
    physically-defensive items against slow-physical-bulky (idx 1).

    Args:
        item: Item name (lowercase).
        archetype_idx: Target archetype cluster index.

    Returns:
        Multiplier >1 for preferred items, <1 for dispreferred, 1.0 for neutral.
    """
    if archetype_idx == 0:
        return 1.5 if item in _SP_DEF_ITEMS else (0.8 if item in _PHYS_DEF_ITEMS else 1.0)
    return 1.5 if item in _PHYS_DEF_ITEMS else (0.8 if item in _SP_DEF_ITEMS else 1.0)


def _spread_counter_multiplier(evs: list[int], archetype_idx: int) -> float:
    """Return a weight multiplier for an EV spread given the target archetype.

    Higher SpD investment is preferred against fast-special (idx 0);
    higher Def investment against slow-physical-bulky (idx 1).

    Args:
        evs: [hp, atk, def, spa, spd, spe] EV values.
        archetype_idx: Target archetype cluster index.

    Returns:
        Multiplier in [0.5, 1.5] based on defensive EV alignment with archetype.
    """
    total_def = evs[2] + evs[4]
    if total_def == 0:
        return 1.0
    if archetype_idx == 0:
        return 0.5 + evs[4] / total_def
    return 0.5 + evs[2] / total_def


def _make_pokemon(
    species: str,
    ability: str,
    item: str,
    nature: str,
    evs: list[int],
    moves: list[str],
) -> Pokemon:
    """Construct a Pokemon object from parsed Smogon components.

    Args:
        species: Species name (lowercase).
        ability: Chosen ability.
        item: Chosen held item.
        nature: Chosen nature.
        evs: [hp, atk, def, spa, spd, spe] EV vector.
        moves: List of chosen move names.

    Returns:
        Populated Pokemon instance.
    """
    return Pokemon(
        species=species,
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


def hydrate_team(
    species_list: list[str],
    archetype_distribution: list[float] | None = None,
    variance: float = 0.20,
    n_moves: int = 4,
) -> list[Pokemon]:
    data = _load_smogmon()
    dominant = _get_dominant_archetype(archetype_distribution)
    team: list[Pokemon] = []

    for species in species_list:
        entry = data.get(species)
        if entry is None:
            team.append(Pokemon(species=species))
            continue

        ability = _weighted_choice(entry.get("Abilities", {}), variance)

        items_pool = entry.get("Items", {})
        if dominant is not None:
            item_mult = {k: _item_counter_multiplier(k, dominant) for k in items_pool}
            item = _weighted_choice(items_pool, variance, counter_mult=item_mult)
        else:
            item = _weighted_choice(items_pool, variance)

        spreads_pool = _filter_valid_spreads(entry.get("Spreads", {}))
        if dominant is not None:
            spread_mult = {}
            for sk in spreads_pool:
                _, evs = _parse_spread(sk)
                spread_mult[sk] = _spread_counter_multiplier(evs, dominant)
            spread_key = _weighted_choice(spreads_pool, variance, counter_mult=spread_mult)
        else:
            spread_key = _weighted_choice(spreads_pool, variance)

        nature, evs = _parse_spread(spread_key)
        moves_pool = entry.get("Moves", {})
        moves = _select_moves(moves_pool, n_moves, variance)

        team.append(_make_pokemon(species, ability, item, nature, evs, moves))

    return team


def _select_moves(
    moves_pool: dict[str, float],
    n: int,
    variance: float,
) -> list[str]:
    """Select the top-n moves from a weighted pool with variance noise.

    Args:
        moves_pool: Mapping of move name → usage weight.
        n: Number of moves to select.
        variance: Noise level injected into each move's selection score.

    Returns:
        List of up to n move names, ordered by adjusted score descending.
    """
    if not moves_pool:
        return []
    items = list(moves_pool.keys())
    if n >= len(items):
        return items[:n]
    weights = [moves_pool[k] for k in items]
    total = sum(weights)
    if total <= 0:
        return items[:n]
    probs = [w / total for w in weights]
    scored = [
        (items[i], probs[i] * (1.0 - variance) + (random.random() * variance))
        for i in range(len(items))
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scored[:n]]
