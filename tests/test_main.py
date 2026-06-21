import numpy as np

from src.agent.base import Pokemon
from src.agent.selection_policy.main import (
    _shannon_entropy,
    _build_team_order,
    _species_showdown_name,
)
from src.agent.selection_policy.double_oracle import Strategy


def test_shannon_entropy_uniform_is_max() -> None:
    probs = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
    h = _shannon_entropy(probs)
    assert np.isclose(h, 2.0, atol=1e-6)


def test_shannon_entropy_certainty_is_zero() -> None:
    probs = np.array([1.0, 0.0], dtype=np.float64)
    h = _shannon_entropy(probs)
    assert np.isclose(h, 0.0, atol=1e-6)


def test_shannon_entropy_single_element() -> None:
    probs = np.array([1.0], dtype=np.float64)
    h = _shannon_entropy(probs)
    assert np.isclose(h, 0.0, atol=1e-6)


def test_shannon_entropy_three_outcomes() -> None:
    probs = np.array([0.5, 0.3, 0.2], dtype=np.float64)
    h = _shannon_entropy(probs)
    assert h > 0.0
    assert h < 2.0


def test_build_team_order_returns_team_string() -> None:
    team = [
        Pokemon(species="snorlax", nature="adamant"),
        Pokemon(species="charizard", nature="timid"),
        Pokemon(species="pikachu", nature="jolly"),
        Pokemon(species="garchomp", nature="jolly"),
        Pokemon(species="landorus", nature="adamant"),
        Pokemon(species="fluttermane", nature="timid"),
    ]
    strategy: Strategy = (0, 1, 2, 3, 4, 5)
    result = _build_team_order(team, strategy)
    assert result.startswith("/team ")
    assert "Snorlax" in result
    assert "Charizard" in result
    assert "Flutter Mane" in result
    parts = result.split("/team ")[1].split("|")
    assert len(parts) == 6


def test_build_team_order_reversed_strategy() -> None:
    team = [
        Pokemon(species="snorlax", nature="adamant"),
        Pokemon(species="charizard", nature="timid"),
        Pokemon(species="pikachu", nature="jolly"),
        Pokemon(species="garchomp", nature="jolly"),
        Pokemon(species="landorus", nature="adamant"),
        Pokemon(species="fluttermane", nature="timid"),
    ]
    strategy: Strategy = (5, 4, 3, 2, 1, 0)
    result = _build_team_order(team, strategy)
    parts = result.split("/team ")[1].split("|")
    assert parts[0] == "Flutter Mane"
    assert parts[-1] == "Snorlax"


def test_species_showdown_name_known() -> None:
    name = _species_showdown_name("landorus")
    assert name == "Landorus"


def test_species_showdown_name_returns_species_for_unknown() -> None:
    name = _species_showdown_name("nonexistent12345")
    assert name == "nonexistent12345"
