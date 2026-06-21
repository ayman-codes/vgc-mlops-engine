import json
import os
import random
from typing import Any

from src.agent.base import Pokemon, Move

SMOGON_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "raw", "smogon", "gen9championsvgc2026regmabo3-1760_2026-05.json")

_smogmon_data: dict[str, dict[str, Any]] | None = None


def _load_smogmon() -> dict[str, dict[str, Any]]:
    global _smogmon_data
    if _smogmon_data is None:
        with open(SMOGON_PATH) as f:
            raw: dict[str, Any] = json.load(f)
        raw_data: dict[str, Any] = raw["data"]
        _smogmon_data = {k.lower(): v for k, v in raw_data.items()}
    return _smogmon_data


def _weighted_choice(options: dict[str, float], variance: float = 0.20) -> str:
    if not options:
        return ""
    items = list(options.keys())
    weights = [options[k] for k in items]
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
    parts = spread_key.split(":")
    nature = parts[0]
    ev_string = parts[1] if len(parts) > 1 else "0/0/0/0/0/0"
    evs = [int(x) for x in ev_string.split("/")]
    return nature, evs


def _make_pokemon(
    species: str,
    ability: str,
    item: str,
    nature: str,
    evs: list[int],
    moves: list[str],
) -> Pokemon:
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
    team: list[Pokemon] = []

    for species in species_list:
        entry = data.get(species)
        if entry is None:
            team.append(Pokemon(species=species))
            continue

        ability = _weighted_choice(entry.get("Abilities", {}), variance)
        item = _weighted_choice(entry.get("Items", {}), variance)
        spread_key = _weighted_choice(entry.get("Spreads", {}), variance)
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
