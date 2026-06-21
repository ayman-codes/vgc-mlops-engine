import pytest

from src.agent.math_utils import calculate_damage, calculate_damage_with_types


def test_calculate_damage_known_value() -> None:
    result = calculate_damage(level=50, power=80, attack=200, defense=100)
    assert result == pytest.approx(66.0, rel=0.5)


def test_flutter_mane_shadow_ball_vs_iron_hands() -> None:
    result = calculate_damage_with_types(
        level=50, power=80, attack=205, defense=110,
        move_type="ghost", defender_types=["fighting", "electric"],
        stab=1.5,
    )
    assert result > 100
