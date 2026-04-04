import pytest
from unittest.mock import MagicMock
from src.agent.battle_policy.heuristics.synergy import calculate_joint_synergy
from src.config.model import BattleWeights

def test_joint_synergy_focus_fire():
    state = MagicMock()
    threat = MagicMock()
    state.sides[1].team.active = [threat, MagicMock()]
    
    weights = BattleWeights(
        W_OFF_DEF_SUPPORT_BONUS=0.0,
        W_BASE_SCORE_A=0.0,
        W_ENV_SYNERGY_BONUS=0.0,
        W_FOCUS_FIRE_BONUS=1.0,
        W_TARGET_PRIORITY_BONUS=1.0,
        W_BASE_SCORE_B=0.0,
        W_SURVIVAL_IMPACT=0.0,
        W_SETUP_SYNERGY_BONUS=0.0
    )
    
    cmd_A = (1, 0)
    cmd_B = (2, 0)
    
    score = calculate_joint_synergy(
        state, cmd_A, cmd_B, biggest_threat=threat, weights=weights, score_A=200.0, score_B=200.0
    )
    
    assert score == 150.0

def test_joint_synergy_off_def_support():
    state = MagicMock()
    weights = BattleWeights(
        W_OFF_DEF_SUPPORT_BONUS=1.0,
        W_BASE_SCORE_A=0.0,
        W_ENV_SYNERGY_BONUS=0.0,
        W_FOCUS_FIRE_BONUS=0.0,
        W_TARGET_PRIORITY_BONUS=0.0,
        W_BASE_SCORE_B=0.0,
        W_SURVIVAL_IMPACT=0.0,
        W_SETUP_SYNERGY_BONUS=0.0
    )
    
    cmd_A = (0, 0)
    cmd_B = (1, 1)
    
    score = calculate_joint_synergy(
        state, cmd_A, cmd_B, biggest_threat=None, weights=weights, score_A=200.0, score_B=200.0
    )
    
    assert score == 75.0

if __name__ == "__main__":
    pytest.main([__file__])