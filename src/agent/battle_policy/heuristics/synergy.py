from typing import Optional
from vgc2.battle_engine import BattleCommand
from vgc2.battle_engine.view import StateView, BattlingPokemonView
from vgc2.battle_engine.modifiers import Weather, Type, Stat
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

    move_A = unit_A.battling_moves[cmd_A[0]].constants if cmd_A[0] < len(unit_A.battling_moves) else None
    move_B = unit_B.battling_moves[cmd_B[0]].constants if cmd_B[0] < len(unit_B.battling_moves) else None

    if not move_A or not move_B:
        return synergy_score

    if target_A == target_B and target_A != -1:
        opponents = state.sides[1].team.active
        target_pkm = opponents[target_A] if target_A < len(opponents) else None
        
        if target_pkm and target_pkm.hp > 0:
            combined_projection = score_A + score_B
            if combined_projection >= target_pkm.hp:
                base_synergy = target_pkm.constants.stats[Stat.MAX_HP] * weights.W_FOCUS_FIRE_BONUS
                synergy_score += base_synergy
                
                if biggest_threat and target_A == opponents.index(biggest_threat):
                    synergy_score += base_synergy * weights.W_TARGET_PRIORITY_BONUS

    if move_A.weather_start != Weather.CLEAR and move_A.weather_start != state.weather:
        if move_A.weather_start == Weather.RAIN and move_B.pkm_type == Type.WATER:
            synergy_score += (score_B * 0.5) * weights.W_ENV_SYNERGY_BONUS
        elif move_A.weather_start == Weather.SUN and move_B.pkm_type == Type.FIRE:
            synergy_score += (score_B * 0.5) * weights.W_ENV_SYNERGY_BONUS

    if move_B.weather_start != Weather.CLEAR and move_B.weather_start != state.weather:
        if move_B.weather_start == Weather.RAIN and move_A.pkm_type == Type.WATER:
            synergy_score += (score_A * 0.5) * weights.W_ENV_SYNERGY_BONUS
        elif move_B.weather_start == Weather.SUN and move_A.pkm_type == Type.FIRE:
            synergy_score += (score_A * 0.5) * weights.W_ENV_SYNERGY_BONUS

    if (move_A.protect and score_B > 0) or (move_B.protect and score_A > 0):
        synergy_score += max(score_A, score_B) * 0.5 * weights.W_OFF_DEF_SUPPORT_BONUS

    return synergy_score