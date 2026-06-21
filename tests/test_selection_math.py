import pytest
from src.agent.selection_policy.math_utils import (
    TYPE_CHART,
    type_effectiveness,
    calculate_damage,
    calculate_damage_with_types,
)


ALL_TYPES = list(TYPE_CHART.keys())


def test_type_chart_coverage_all_types_present() -> None:
    assert len(TYPE_CHART) == 18


def test_type_chart_electric_immune_to_ground() -> None:
    assert type_effectiveness("electric", "ground") == 0.0


def test_type_chart_fighting_super_effective_vs_normal() -> None:
    assert type_effectiveness("fighting", "normal") == 2.0


def test_type_chart_fire_resists_fire() -> None:
    assert type_effectiveness("fire", "fire") == 0.5


def test_type_chart_unknown_move_type_defaults_neutral() -> None:
    assert type_effectiveness("nonexistent", "normal") == 1.0


def test_type_chart_unknown_defender_type_defaults_neutral() -> None:
    assert type_effectiveness("fire", "nonexistent") == 1.0


def test_type_chart_normal_immune_to_ghost_attacks() -> None:
    assert type_effectiveness("ghost", "normal") == 0.0


def test_type_chart_all_types_have_defender_mappings() -> None:
    defender_types: set[str] = set()
    for move_type, defenders in TYPE_CHART.items():
        defender_types.update(defenders.keys())
    for t in ALL_TYPES:
        assert t in defender_types, f"{t} missing as defender in all entries"


def test_calculate_damage_known_value() -> None:
    result = calculate_damage(level=50, power=80, attack=200, defense=100, modifier=1.0)
    assert result == pytest.approx(66.0, rel=0.5)


def test_calculate_damage_high_defense() -> None:
    result = calculate_damage(level=50, power=120, attack=150, defense=200, modifier=1.0)
    assert result > 0.0
    assert result < 50.0


def test_calculate_damage_modifier_scales() -> None:
    base = calculate_damage(level=50, power=80, attack=200, defense=100, modifier=1.0)
    doubled = calculate_damage(level=50, power=80, attack=200, defense=100, modifier=2.0)
    assert doubled == pytest.approx(2.0 * base, rel=1e-9)


def test_calculate_damage_modifier_can_reduce() -> None:
    base = calculate_damage(level=50, power=80, attack=200, defense=100, modifier=1.0)
    halved = calculate_damage(level=50, power=80, attack=200, defense=100, modifier=0.5)
    assert halved == pytest.approx(0.5 * base, rel=1e-9)


def test_calculate_damage_minimum_one() -> None:
    result = calculate_damage(level=1, power=10, attack=10, defense=500, modifier=1.0)
    assert result >= 1.0


def test_flutter_mane_shadow_ball_vs_iron_hands() -> None:
    damage = calculate_damage_with_types(
        level=83,
        power=80,
        attack=205,
        defense=110,
        move_type="ghost",
        defender_types=["fighting", "electric"],
        stab=1.5,
    )
    assert damage > 100.0


def test_no_stab_no_effectiveness() -> None:
    damage = calculate_damage_with_types(
        level=50,
        power=60,
        attack=100,
        defense=100,
        move_type="normal",
        defender_types=["normal"],
        stab=1.0,
    )
    expected = calculate_damage(50, 60, 100, 100, 1.0)
    assert damage == pytest.approx(expected, rel=1e-9)


def test_quadruple_resisted() -> None:
    eff = type_effectiveness("grass", "steel") * type_effectiveness("grass", "fire")
    assert eff == pytest.approx(0.25, rel=1e-9)


def test_quadruple_super_effective() -> None:
    eff = type_effectiveness("ice", "dragon") * type_effectiveness("ice", "flying")
    assert eff == pytest.approx(4.0, rel=1e-9)


def test_all_type_effectiveness_values_valid() -> None:
    for move_type, defenders in TYPE_CHART.items():
        for defender_type, multiplier in defenders.items():
            assert multiplier in {0.0, 0.5, 2.0}, f"{move_type} vs {defender_type}: {multiplier}"
