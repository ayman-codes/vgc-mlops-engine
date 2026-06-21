import numpy as np

from src.agent.base import Pokemon, Move
from src.agent.selection_policy.double_oracle import (
    find_best_response,
    enumerate_top_k_strategies,
    _pokemon_to_showdown,
    _nature_multipliers,
    _species_types,
    _pokemon_stats,
)


def test_find_best_response_maximizes_expected_payoff() -> None:
    matrix = np.array([[0.6, 0.4], [0.3, 0.7]], dtype=np.float64)
    opp_mix = np.array([0.5, 0.5], dtype=np.float64)
    idx = find_best_response(matrix, opp_mix)
    expected0 = 0.6 * 0.5 + 0.4 * 0.5
    expected1 = 0.3 * 0.5 + 0.7 * 0.5
    expected_best = 0 if expected0 >= expected1 else 1
    assert idx == expected_best


def test_find_best_response_one_dominant() -> None:
    matrix = np.array([[0.9, 0.8], [0.2, 0.1]], dtype=np.float64)
    opp_mix = np.array([0.5, 0.5], dtype=np.float64)
    idx = find_best_response(matrix, opp_mix)
    assert idx == 0


def test_find_best_response_single_strategy() -> None:
    matrix = np.array([[0.5]], dtype=np.float64)
    opp_mix = np.array([1.0], dtype=np.float64)
    idx = find_best_response(matrix, opp_mix)
    assert idx == 0


def test_nature_multipliers_known() -> None:
    mult = _nature_multipliers("modest")
    assert mult["spa"] == 1.1
    assert mult["atk"] == 0.9


def test_nature_multipliers_unknown_defaults_empty() -> None:
    mult = _nature_multipliers("nonexistent")
    assert mult == {}


def test_nature_multipliers_case_insensitive() -> None:
    mult = _nature_multipliers("ADAMANT")
    assert mult["atk"] == 1.1


def test_species_types_known() -> None:
    types = _species_types("charizard")
    assert "fire" in types
    assert "flying" in types


def test_species_types_unknown_defaults_normal() -> None:
    types = _species_types("nonexistentspecies12345")
    assert types == ["normal"]


def test_pokemon_to_showdown_contains_species() -> None:
    mon = Pokemon(species="pikachu", ev_hp=252, ev_atk=252, nature="adamant")
    text = _pokemon_to_showdown(mon)
    assert "Pikachu" in text
    assert "Adamant Nature" in text


def test_pokemon_to_showdown_includes_item() -> None:
    mon = Pokemon(species="snorlax", item="leftovers", nature="careful")
    text = _pokemon_to_showdown(mon)
    assert "Leftovers" in text or "leftovers" in text


def test_pokemon_to_showdown_includes_moves() -> None:
    mon = Pokemon(
        species="snorlax",
        moves=[Move(name="bodyslam"), Move(name="earthquake")],
        nature="adamant",
    )
    text = _pokemon_to_showdown(mon)
    assert "Body Slam" in text or "Bodyslam" in text


def test_pokemon_stats_basic() -> None:
    mon = Pokemon(species="snorlax", ev_hp=252, ev_atk=252, nature="adamant", level=50)
    stats = _pokemon_stats(mon)
    assert "hp" in stats
    assert "atk" in stats
    assert stats["atk"] > 0


def test_enumerate_top_k_strategies_needs_six() -> None:
    team = [Pokemon(species="snorlax") for _ in range(4)]
    result = enumerate_top_k_strategies(team, k=3)
    assert result == []


def test_enumerate_top_k_strategies_returns_strategies() -> None:
    team = [
        Pokemon(
            species="snorlax",
            moves=[Move(name="bodyslam"), Move(name="earthquake")],
            nature="adamant", ev_hp=252, ev_atk=252,
        ) for _ in range(6)
    ]
    result = enumerate_top_k_strategies(team, k=3)
    assert len(result) == 3
    for s in result:
        assert len(s) == 6
        assert len(set(s)) == 6


def test_enumerate_top_k_strategies_k_equals_one() -> None:
    team = [
        Pokemon(
            species="snorlax",
            moves=[Move(name="bodyslam")],
            nature="adamant", ev_hp=252, ev_atk=252,
        ) for _ in range(6)
    ]
    result = enumerate_top_k_strategies(team, k=1)
    assert len(result) == 1
