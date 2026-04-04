import pytest
from unittest.mock import MagicMock, patch
from src.agent.battle_policy.main import MyBattlePolicy

@patch('src.agent.battle_policy.main.load_battle_weights')
@patch('src.agent.battle_policy.main._score_single_offensive_move')
@patch('src.agent.battle_policy.main._identify_biggest_threat_opponent')
def test_decision_joint_resolution(mock_identify_threat, mock_score_move, mock_load_weights):
    mock_weights = MagicMock()
    mock_weights.W_BASE_SCORE_A = 1.0
    mock_weights.W_BASE_SCORE_B = 1.0
    mock_weights.W_FOCUS_FIRE_BONUS = 0.0
    mock_weights.W_TARGET_PRIORITY_BONUS = 0.0
    mock_weights.W_OFF_DEF_SUPPORT_BONUS = 0.0
    mock_load_weights.return_value = mock_weights
    
    mock_score_move.return_value = 500.0
    mock_identify_threat.return_value = (None, 0.0, False, 1.0)

    policy = MyBattlePolicy()
    state = MagicMock()
    
    pkm_A = MagicMock()
    pkm_A.hp = 100
    move_A = MagicMock()
    move_A.constants.protect = False
    pkm_A.battling_moves = [move_A]
    
    pkm_B = MagicMock()
    pkm_B.hp = 100
    move_B = MagicMock()
    move_B.constants.protect = False
    pkm_B.battling_moves = [move_B]
    
    opp = MagicMock()
    opp.hp = 100
    
    side_0 = MagicMock()
    side_0.team.active = [pkm_A, pkm_B]
    side_0.team.reserve = []
    
    side_1 = MagicMock()
    side_1.team.active = [opp]
    
    state.sides = [side_0, side_1]
    
    policy.battle_params = MagicMock()
    
    commands = policy.decision(state, turn_count=1)
    
    assert len(commands) == 2, f"Evaluation collapse. Returned commands: {commands}"
    assert commands[0] == (0, 0)
    assert commands[1] == (0, 0)

    policy.detailed_logging = True
    policy.decision(state, turn_count=2)
    telemetry = policy.get_telemetry()
    
    assert telemetry["turn_index"] == 2
    assert "slot_0" in telemetry
    assert "joint_q_score" in telemetry
if __name__ == "__main__":
    pytest.main([__file__])