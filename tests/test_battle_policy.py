import pytest
from unittest.mock import MagicMock, patch
from src.agent.battle_policy.heuristics.threat import calculate_effective_speed, estimate_incoming_threat
from vgc2.battle_engine.modifiers import Stat, Type

def create_mock_pokemon(speed: int, hp: int = 100) -> MagicMock:
    mock_pkm = MagicMock()
    mock_pkm.constants.stats = {Stat.SPEED: speed, Stat.MAX_HP: 100}
    mock_pkm.constants.types = [Type.NORMAL]
    mock_pkm.boosts = {Stat.SPEED: 0}
    mock_pkm.hp = hp
    mock_pkm.battling_moves = []
    return mock_pkm

def test_speed_resolution_tailwind():
    pkm = create_mock_pokemon(speed=100)
    state = MagicMock()
    state.sides[0].conditions.tailwind = True
    assert calculate_effective_speed(pkm, state, side_index=0) == 200

@patch('src.agent.battle_policy.heuristics.threat.calculate_damage')
@patch('src.agent.battle_policy.heuristics.threat.get_type_multiplier')
def test_lethal_threat_penalty(mock_get_type, mock_calc_damage):
    mock_calc_damage.return_value = 200.0
    mock_get_type.return_value = 2.0
    
    unit = create_mock_pokemon(speed=50, hp=10)
    opp = create_mock_pokemon(speed=100)
    
    mock_move = MagicMock()
    opp.battling_moves = [mock_move]
    
    state = MagicMock()
    state.trickroom = False
    state.sides[0].conditions.tailwind = False
    state.sides[1].conditions.tailwind = False
    state.sides[1].team.active = [opp]
    
    params = MagicMock()
    threat = estimate_incoming_threat(unit, unit_side=0, state=state, params=params)
    
    assert threat["is_lethal"] is True
    assert threat["is_outsped"] is True
    assert threat["penalty_multiplier"] == 0.01
    assert threat["aggro_score"] == 4.0

if __name__ == "__main__":
    pytest.main([__file__])