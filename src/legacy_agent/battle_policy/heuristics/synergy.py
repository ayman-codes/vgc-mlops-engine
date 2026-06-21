from typing import Optional, Any
from poke_env.battle import Pokemon, AbstractBattle

class DummyWeights:
    W_FOCUS_FIRE_BONUS = 0.5
    W_TARGET_PRIORITY_BONUS = 0.3
    W_ENV_SYNERGY_BONUS = 0.2
    W_OFF_DEF_SUPPORT_BONUS = 0.4
    W_TYPE_COVERAGE_BONUS = 0.3
    W_SPEED_CONTROL_BONUS = 0.25
    W_WEATHER_TERRAIN_BONUS = 0.2
    W_BASE_SCORE_A = 1.0
    W_BASE_SCORE_B = 1.0

def _get_type_name(pokemon: Pokemon, idx: int = 0) -> str:
    if hasattr(pokemon, 'types') and len(pokemon.types) > idx:
        t = pokemon.types[idx]
        return getattr(t, 'name', str(t)).lower() if hasattr(t, 'name') else str(t).lower()
    return 'normal'

def _calculate_type_coverage(unit_a: Pokemon, unit_b: Pokemon) -> float:
    types_a = set(_get_type_name(unit_a, i) for i in range(len(unit_a.types)))
    types_b = set(_get_type_name(unit_b, i) for i in range(len(unit_b.types)))
    
    unique_types = types_a | types_b
    overlap = types_a & types_b
    
    coverage_score = len(unique_types) * 0.5
    overlap_penalty = len(overlap) * 0.3
    
    return float(coverage_score - overlap_penalty)

def calculate_joint_synergy(
    state: AbstractBattle,
    cmd_A: Any,
    cmd_B: Any,
    unit_A: Pokemon,
    unit_B: Pokemon,
    biggest_threat: Optional[Pokemon],
    weights: Any,
    score_A: float,
    score_B: float
) -> float:
    synergy_score = 0.0
    
    target_A = cmd_A[1] if isinstance(cmd_A, tuple) else None
    target_B = cmd_B[1] if isinstance(cmd_B, tuple) else None
    
    move_A = cmd_A[0] if isinstance(cmd_A, tuple) else cmd_A
    move_B = cmd_B[0] if isinstance(cmd_B, tuple) else cmd_B

    if target_A is not None and target_B is not None and target_A == target_B:
        synergy_score += (score_A + score_B) * weights.W_FOCUS_FIRE_BONUS
        
        if biggest_threat and biggest_threat.active:
            synergy_score += (score_A + score_B) * weights.W_TARGET_PRIORITY_BONUS

    if hasattr(move_A, 'id') and move_A.id == 'protect' and score_B > 0:
        synergy_score += score_B * weights.W_OFF_DEF_SUPPORT_BONUS
    if hasattr(move_B, 'id') and move_B.id == 'protect' and score_A > 0:
        synergy_score += score_A * weights.W_OFF_DEF_SUPPORT_BONUS

    synergy_score += _calculate_type_coverage(unit_A, unit_B) * weights.W_TYPE_COVERAGE_BONUS

    speed_control_moves = {'tailwind', 'thunderwave', 'icywind', 'electroweb', 'rocktomb', 'stringshot', 'stickyweb', 'trickroom'}
    has_speed_control = (hasattr(move_A, 'id') and move_A.id in speed_control_moves) or \
                        (hasattr(move_B, 'id') and move_B.id in speed_control_moves)
    if has_speed_control and (score_A > 0 or score_B > 0):
        synergy_score += max(score_A, score_B) * weights.W_SPEED_CONTROL_BONUS

    weather_moves = {'sunnyday', 'raindance', 'sandstorm', 'hail', 'snow'}
    terrain_moves = {'electricterrain', 'grassyterrain', 'mistyterrain', 'psychicterrain'}
    sets_weather = (hasattr(move_A, 'id') and move_A.id in weather_moves) or \
                   (hasattr(move_B, 'id') and move_B.id in weather_moves)
    sets_terrain = (hasattr(move_A, 'id') and move_A.id in terrain_moves) or \
                   (hasattr(move_B, 'id') and move_B.id in terrain_moves)
    
    if sets_weather or sets_terrain:
        synergy_score += (score_A + score_B) * weights.W_WEATHER_TERRAIN_BONUS

    return float(synergy_score)
