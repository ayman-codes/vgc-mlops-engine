import pytest
from unittest.mock import MagicMock
from src.agent.battle_policy.heuristics.threat import calculate_effective_speed, estimate_incoming_threat
from src.agent.battle_policy.heuristics.synergy import calculate_joint_synergy
from vgc2.battle_engine.modifiers import Stat
from src.config.model import BattleWeights

def create_mock_pokemon(speed: int, hp: int = 100) -> MagicMock:
    mock_pkm = MagicMock()
    mock_pkm.constants.stats = {Stat.SPEED: speed, Stat.MAX_HP: 100}
    mock_pkm.boosts = {Stat.SPEED: 0}
    mock_pkm.hp = hp
    mock_pkm.battling_moves = []
    return mock_pkm

def test_speed_resolution_tailwind():
    pkm = create_mock_pokemon(speed=100)
    state = MagicMock()
    state.sides[0].conditions.tailwind = True
    assert calculate_effective_speed(pkm, state, side_index=0) == 200

def test_lethal_threat_penalty():
    unit = create_mock_pokemon(speed=50, hp=10)
    opp = create_mock_pokemon(speed=100)
    mock_move = MagicMock()
    mock_move.constants.base_power = 200
    opp.battling_moves = [mock_move]
    
    state = MagicMock()
    state.trickroom = False
    state.sides[0].conditions.tailwind = False
    state.sides[1].conditions.tailwind = False
    state.sides[1].team.active = [opp]
    
    threat = estimate_incoming_threat(unit, unit_side=0, state=state)
    assert threat["is_lethal"] is True
    assert threat["is_outsped"] is True
    assert threat["penalty_multiplier"] == 0.01

def test_joint_synergy_focus_fire():
    state = MagicMock()
    weights = BattleWeights(
        W_OFF_DEF_SUPPORT_BONUS=0.0,
        W_BASE_SCORE_A=0.0,
        W_ENV_SYNERGY_BONUS=0.0,
        W_FOCUS_FIRE_BONUS=1.0,
        W_TARGET_PRIORITY_BONUS=0.0,
        W_BASE_SCORE_B=0.0,
        W_SURVIVAL_IMPACT=0.0,
        W_SETUP_SYNERGY_BONUS=0.0
    )
    cmd_A = (1, 0)
    cmd_B = (2, 0)
    
    score = calculate_joint_synergy(state, cmd_A, cmd_B, None, weights, 200.0, 200.0)
    assert score == 100.0

if __name__ == "__main__":
    pytest.main([__file__])