"""Double Oracle strategy enumeration and payoff matrix expansion.

Scores team lead/back permutations via Gen 9 damage formula and
expands the payoff matrix through Showdown battles.
"""

import asyncio
import math
from typing import Any, Callable
import numpy as np
from numpy.typing import NDArray

from src.agent.base import Pokemon
from src.agent.selection_policy.math_utils import calculate_damage_with_types

Strategy = tuple[int, int, int, int, int, int]

_GEN_DATA_CACHE: Any = None


def _gen_data() -> Any:
    global _GEN_DATA_CACHE
    if _GEN_DATA_CACHE is None:
        from poke_env.data import GenData
        _GEN_DATA_CACHE = GenData.from_gen(9)
    return _GEN_DATA_CACHE


def _get_pokedex_entry(species: str) -> Any:
    return _gen_data().pokedex.get(species)


def _get_move_data(move_name: str) -> Any:
    return _gen_data().moves.get(move_name)


def _pokemon_stats(mon: Pokemon) -> dict[str, float]:
    dex = _get_pokedex_entry(mon.species) or {}
    base = dex.get("baseStats", {})
    level = mon.level
    hp_val = math.floor((2 * base.get("hp", 100) + mon.iv_hp + mon.ev_hp / 4) * level / 100 + level + 10)
    atk_val = math.floor((2 * base.get("atk", 100) + mon.iv_atk + mon.ev_atk / 4) * level / 100 + 5)
    def_val = math.floor((2 * base.get("def", 100) + mon.iv_def + mon.ev_def / 4) * level / 100 + 5)
    spa_val = math.floor((2 * base.get("spa", 100) + mon.iv_spa + mon.ev_spa / 4) * level / 100 + 5)
    spd_val = math.floor((2 * base.get("spd", 100) + mon.iv_spd + mon.ev_spd / 4) * level / 100 + 5)
    spe_val = math.floor((2 * base.get("spe", 100) + mon.iv_spe + mon.ev_spe / 4) * level / 100 + 5)

    nature_mult = _nature_multipliers(mon.nature)
    for stat_name, mult in nature_mult.items():
        if stat_name == "atk":
            atk_val = math.floor(atk_val * mult)
        elif stat_name == "def":
            def_val = math.floor(def_val * mult)
        elif stat_name == "spa":
            spa_val = math.floor(spa_val * mult)
        elif stat_name == "spd":
            spd_val = math.floor(spd_val * mult)
        elif stat_name == "spe":
            spe_val = math.floor(spe_val * mult)

    return {
        "hp": hp_val, "atk": atk_val, "def": def_val,
        "spa": spa_val, "spd": spd_val, "spe": spe_val,
    }


def _nature_multipliers(nature: str) -> dict[str, float]:
    boosts: dict[str, dict[str, float]] = {
        "adamant": {"atk": 1.1, "spa": 0.9},
        "bold": {"def": 1.1, "atk": 0.9},
        "brave": {"atk": 1.1, "spe": 0.9},
        "calm": {"spd": 1.1, "atk": 0.9},
        "careful": {"spd": 1.1, "spa": 0.9},
        "gentle": {"spd": 1.1, "def": 0.9},
        "hasty": {"spe": 1.1, "def": 0.9},
        "impish": {"def": 1.1, "spa": 0.9},
        "jolly": {"spe": 1.1, "spa": 0.9},
        "lax": {"def": 1.1, "spd": 0.9},
        "lonely": {"atk": 1.1, "def": 0.9},
        "mild": {"spa": 1.1, "def": 0.9},
        "modest": {"spa": 1.1, "atk": 0.9},
        "naive": {"spe": 1.1, "spd": 0.9},
        "naughty": {"atk": 1.1, "spd": 0.9},
        "quiet": {"spa": 1.1, "spe": 0.9},
        "rash": {"spa": 1.1, "spd": 0.9},
        "relaxed": {"def": 1.1, "spe": 0.9},
        "sassy": {"spd": 1.1, "spe": 0.9},
        "timid": {"spe": 1.1, "atk": 0.9},
    }
    return boosts.get(nature.lower(), {})


def _species_types(species: str) -> list[str]:
    dex = _get_pokedex_entry(species)
    if dex is None:
        return ["normal"]
    return [t.lower() for t in dex.get("types", ["normal"])]


def _score_lead_pair(
    lead1: Pokemon,
    lead2: Pokemon,
    opponent_types: list[str],
) -> float:
    dmg_sum = 0.0
    for lead in (lead1, lead2):
        stats = _pokemon_stats(lead)
        mon_types = _species_types(lead.species)
        best_dmg = 0.0
        for move in lead.moves:
            move_data = _get_move_data(move.name)
            if move_data is None:
                continue
            bp = move_data.get("basePower", 0)
            if bp <= 0 or bp is None:
                continue
            move_type = move_data.get("type", "normal").lower()
            category = move_data.get("category", "physical")
            if category.startswith("physical"):
                attack = stats["atk"]
                defense = 80
            else:
                attack = stats["spa"]
                defense = 80
            stab = 1.5 if move_type in mon_types else 1.0
            dmg = calculate_damage_with_types(
                50, int(bp), int(attack), defense,
                move_type, opponent_types, stab,
            )
            best_dmg = max(best_dmg, dmg)
        dmg_sum += best_dmg
    return dmg_sum


def enumerate_top_k_strategies(
    team: list[Pokemon],
    k: int = 3,
    opponent_types: list[str] | None = None,
) -> list[Strategy]:
    """Return top-K full-team orderings by lead damage output.

    Args:
        team: List of 6 hydrated Pokemon (must have exactly 6).
        k: Number of strategies to return (clamped to available pairs).
        opponent_types: Target type list for damage estimation;
            defaults to ["normal"].

    Returns:
        List of (lead1, lead2, back1, back2, back3, back4) index tuples
        ordered by lead-pair damage score descending.
    """
    if len(team) != 6:
        return []
    target = opponent_types or ["normal"]
    indices = list(range(6))
    scored_pairs: list[tuple[float, tuple[int, int]]] = []
    for i in range(6):
        for j in range(6):
            if i == j:
                continue
            dmg = _score_lead_pair(team[i], team[j], target)
            scored_pairs.append((dmg, (i, j)))
    scored_pairs.sort(key=lambda x: x[0], reverse=True)
    top_pairs = [p[1] for p in scored_pairs[:min(k, len(scored_pairs))]]
    strategies: list[Strategy] = []
    for lead1, lead2 in top_pairs:
        back = [idx for idx in indices if idx not in (lead1, lead2)]
        s: Strategy = (lead1, lead2, back[0], back[1], back[2], back[3])
        strategies.append(s)
    return strategies


def find_best_response(
    payoff_matrix: NDArray[np.float64],
    opponent_mix: NDArray[np.float64],
) -> int:
    """Return the index of our strategy with highest expected payoff.

    Args:
        payoff_matrix: Shape (our_strategies, opponent_strategies) — our win rate.
        opponent_mix: Probability vector over opponent strategies.

    Returns:
        Index into our_strategies with maximum expected payoff.
    """
    expected = payoff_matrix @ opponent_mix
    return int(np.argmax(expected))


def _move_showdown_name(move_name: str) -> str:
    move_data = _get_move_data(move_name)
    if move_data:
        name: str = move_data.get("name", move_name)
        return name
    return move_name.replace("-", " ").title()


def _species_showdown_name(species: str) -> str:
    dex = _get_pokedex_entry(species)
    if dex:
        name: str = dex.get("name", species)
        return name
    return species


def _pokemon_to_showdown(mon: Pokemon) -> str:
    line = _species_showdown_name(mon.species)
    if mon.item:
        item_name = mon.item.replace("_", " ").title()
        line += f" @ {item_name}"
    lines = [line]
    if mon.ability:
        ability_name = mon.ability.replace("_", " ").title()
        lines.append(f"Ability: {ability_name}")
    lines.append(f"Level: {mon.level}")
    ev_parts = []
    for ev_val, ev_label in [
        (mon.ev_hp, "HP"), (mon.ev_atk, "Atk"), (mon.ev_def, "Def"),
        (mon.ev_spa, "SpA"), (mon.ev_spd, "SpD"), (mon.ev_spe, "Spe"),
    ]:
        if ev_val > 0:
            ev_parts.append(f"{ev_val} {ev_label}")
    if ev_parts:
        lines.append("EVs: " + " / ".join(ev_parts))
    if mon.nature:
        nature_str = mon.nature.capitalize()
        if not nature_str.endswith("Nature"):
            nature_str += " Nature"
        lines.append(nature_str)
    for move in mon.moves:
        lines.append(f"- {_move_showdown_name(move.name)}")
    return "\n".join(lines)


async def _showdown_battle(
    our_team: list[Pokemon],
    our_strategy: Strategy,
    opponent_team: list[Pokemon],
    opponent_strategy: Strategy,
    server_url: str = "http://localhost:8000",
    battle_format: str = "gen9randombattle",
) -> float:
    from poke_env import AccountConfiguration, ServerConfiguration
    from poke_env.player import RandomPlayer

    config = ServerConfiguration(server_url, server_url)
    tag = f"{np.random.randint(10000, 99999)}"

    showdown_our = "\n\n".join(
        _pokemon_to_showdown(our_team[i]) for i in our_strategy
    )
    showdown_opp = "\n\n".join(
        _pokemon_to_showdown(opponent_team[i]) for i in opponent_strategy
    )

    us = RandomPlayer(
        account_configuration=AccountConfiguration(f"oracle-us-{tag}", ""),
        server_configuration=config,
        battle_format=battle_format,
        team=showdown_our,
    )
    opp = RandomPlayer(
        account_configuration=AccountConfiguration(f"oracle-opp-{tag}", ""),
        server_configuration=config,
        battle_format=battle_format,
        team=showdown_opp,
    )

    try:
        await us.battle_against(opp, n_battles=1)
        won = us.n_won_battles > 0
        return 1.0 if won else 0.0
    except Exception:
        return 0.5


async def expand_matrix_async(
    our_strategies: list[Strategy],
    opponent_strategies: list[Strategy],
    our_team: list[Pokemon],
    opponent_team: list[Pokemon],
    batch_size: int = 4,
    battle_fn: Callable[..., Any] | None = None,
) -> NDArray[np.float64]:
    """Compute payoff matrix for all strategy pairs via Showdown battles.

    Args:
        our_strategies: List of strategies (6-element index tuples) for our team.
        opponent_strategies: List of strategies for the opponent team.
        our_team: Our 6 Pokemon.
        opponent_team: Opponent's 6 Pokemon.
        batch_size: Number of concurrent battles (unused if battle_fn is None).
        battle_fn: Async function to evaluate one strategy pair;
            defaults to _showdown_battle.

    Returns:
        Float64 matrix of shape (our_strategies, opponent_strategies) with win rates.
    """
    n = len(our_strategies)
    m = len(opponent_strategies)
    matrix = np.zeros((n, m), dtype=np.float64)
    fn = battle_fn or _showdown_battle
    for i in range(n):
        row_results = await asyncio.gather(*[
            fn(our_team, our_strategies[i], opponent_team, opponent_strategies[j])
            for j in range(m)
        ])
        for j, result in enumerate(row_results):
            matrix[i, j] = result
    return matrix
