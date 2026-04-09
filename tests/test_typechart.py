import pytest
from vgc2.battle_engine.modifiers import Type
from src.agent.battle_policy.utils.type_chart import get_type_multiplier

def test_single_type_super_effective():
    assert get_type_multiplier(Type.FIRE, Type.GRASS) == 2.0

def test_single_type_not_very_effective():
    assert get_type_multiplier(Type.WATER, Type.WATER) == 0.5

def test_single_type_immunity():
    assert get_type_multiplier(Type.NORMAL, Type.GHOST) == 0.0

def test_dual_type_double_weakness():
    assert get_type_multiplier(Type.FIGHT, Type.NORMAL, Type.ROCK) == 4.0

def test_dual_type_double_resistance():
    assert get_type_multiplier(Type.WATER, Type.GRASS, Type.DRAGON) == 0.25

def test_dual_type_immunity_override():
    assert get_type_multiplier(Type.GROUND, Type.POISON, Type.FLYING) == 0.0
    
def test_missing_secondary_type():
    assert get_type_multiplier(Type.ELECTRIC, Type.WATER, None) == 2.0

if __name__ == "__main__":
    pytest.main([__file__])