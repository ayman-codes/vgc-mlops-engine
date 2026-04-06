from typing import Optional
from vgc2.battle_engine import BattleCommand
from vgc2.battle_engine.view import StateView, BattlingPokemonView
from src.config.model import BattleWeights

def calculate_joint_synergy(
    state: StateView,
    cmd_A: BattleCommand,
    cmd_B: BattleCommand,
    unit_A: BattlingPokemonView,
    unit_B: BattlingPokemonView,
    biggest_threat: Optional[BattlingPokemonView],
    weights: BattleWeights,
    score_A: float,
    score_B: float
) -> float:
    synergy_score = 0.0
    
    is_move_A = cmd_A[0] >= 0
    is_move_B = cmd_B[0] >= 0
    
    if not is_move_A or not is_move_B:
        return synergy_score

    target_A = cmd_A[1]
    target_B = cmd_B[1]

    if target_A == target_B and target_A != -1:
        if score_A > 0 and score_B > 0:
            synergy_score += (100.0 * weights.W_FOCUS_FIRE_BONUS)
            if biggest_threat and target_A == state.sides[1].team.active.index(biggest_threat):
                synergy_score += (50.0 * weights.W_TARGET_PRIORITY_BONUS)

    is_protect_A = False
    if is_move_A and cmd_A[0] < len(unit_A.battling_moves):
        is_protect_A = unit_A.battling_moves[cmd_A[0]].constants.protect

    is_protect_B = False
    if is_move_B and cmd_B[0] < len(unit_B.battling_moves):
        is_protect_B = unit_B.battling_moves[cmd_B[0]].constants.protect
    
    if (is_protect_A and score_B > 150) or (is_protect_B and score_A > 150):
        synergy_score += (75.0 * weights.W_OFF_DEF_SUPPORT_BONUS)

    return synergy_score