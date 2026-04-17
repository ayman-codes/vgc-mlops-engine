from unittest.mock import MagicMock
from vgc2.battle_engine.modifiers import Weather, Stat
from src.agent.battle_policy.heuristics.synergy import calculate_joint_synergy

def test_joint_synergy_focus_fire() -> None:
    state = MagicMock()
    target_pkm = MagicMock()
    target_pkm.hp = 100.0
    target_pkm.constants.stats = {Stat.MAX_HP: 100.0}
    
    opponents = [target_pkm]
    state.sides = {1: MagicMock(team=MagicMock(active=opponents))}
    
    cmd_A = (0, 0)
    cmd_B = (0, 0)
    
    unit_A = MagicMock()
    unit_B = MagicMock()
    
    move_A = MagicMock()
    move_A.weather_start = Weather.CLEAR
    move_A.protect = False
    unit_A.battling_moves = [MagicMock(constants=move_A)]
    
    move_B = MagicMock()
    move_B.weather_start = Weather.CLEAR
    move_B.protect = False
    unit_B.battling_moves = [MagicMock(constants=move_B)]
    
    weights = MagicMock()
    weights.W_FOCUS_FIRE_BONUS = 0.5
    weights.W_TARGET_PRIORITY_BONUS = 0.0
    weights.W_OFF_DEF_SUPPORT_BONUS = 0.0
    
    score = calculate_joint_synergy(state, cmd_A, cmd_B, unit_A, unit_B, None, weights, 60.0, 50.0)
    
    assert score == 50.0

def test_joint_synergy_off_def_support() -> None:
    state = MagicMock()
    state.sides = {1: MagicMock(team=MagicMock(active=[MagicMock(hp=100.0), MagicMock(hp=100.0)]))}
    
    cmd_A = (0, -1) 
    cmd_B = (0, 1)
    
    unit_A = MagicMock()
    unit_B = MagicMock()
    
    move_A = MagicMock()
    move_A.protect = True
    move_A.weather_start = Weather.CLEAR
    unit_A.battling_moves = [MagicMock(constants=move_A)]
    
    move_B = MagicMock()
    move_B.protect = False
    move_B.weather_start = Weather.CLEAR
    unit_B.battling_moves = [MagicMock(constants=move_B)]
    
    weights = MagicMock()
    weights.W_OFF_DEF_SUPPORT_BONUS = 2.0
    
    score = calculate_joint_synergy(state, cmd_A, cmd_B, unit_A, unit_B, None, weights, 0.0, 100.0)
    
    assert score == 100.0