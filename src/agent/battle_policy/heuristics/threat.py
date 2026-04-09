from typing import List, Tuple, Optional, Dict, Any
from vgc2.battle_engine.view import StateView, BattlingPokemonView
from vgc2.battle_engine.modifiers import Stat, Type
from vgc2.battle_engine.constants import BattleRuleParam
from vgc2.battle_engine.damage_calculator import calculate_damage
from src.agent.battle_policy.utils.type_chart import get_type_multiplier

def calculate_effective_speed(pkm: BattlingPokemonView, state: StateView, side_index: int) -> int:
    base_speed = pkm.constants.stats[Stat.SPEED]
    stage = pkm.boosts[Stat.SPEED]
    
    if stage >= 0:
        multiplier = (2 + stage) / 2
    else:
        multiplier = 2 / (2 - stage)
    
    speed = int(base_speed * multiplier)
    if state.sides[side_index].conditions.tailwind:
        speed *= 2
    return speed

def _identify_biggest_threat_opponent(
    unit: BattlingPokemonView, 
    unit_side: int, 
    opponents: List[BattlingPokemonView], 
    state: StateView,
    params: BattleRuleParam
) -> Tuple[Optional[BattlingPokemonView], float, bool, float]:
    max_dmg = 0.0
    max_type_mult = 1.0
    primary_threat = None
    
    valid_opponents = [opp for opp in opponents if opp and opp.hp > 0]
    if not valid_opponents:
        return None, 0.0, False, 1.0

    primary_threat = valid_opponents[0]
    unit_speed = calculate_effective_speed(unit, state, unit_side)
    trick_room = state.trickroom

    u_types = unit.constants.species.types
    def_type_1 = u_types[0] if len(u_types) > 0 else Type.NORMAL
    def_type_2 = u_types[1] if len(u_types) > 1 else None

    for opp in valid_opponents:
        current_opp_max_dmg = 0.0
        current_opp_max_mult = 1.0
        if opp.battling_moves:
            for move in opp.battling_moves:
                dmg = calculate_damage(
                    params=params,
                    attacking_side=1 - unit_side,
                    move=move.constants,
                    state=state,
                    attacker=opp,
                    defender=unit
                )
                atk_type = move.constants.pkm_type
                type_mult = get_type_multiplier(atk_type, def_type_1, def_type_2)
                
                if dmg > current_opp_max_dmg:
                    current_opp_max_dmg = dmg
                if type_mult > current_opp_max_mult:
                    current_opp_max_mult = type_mult
        
        if current_opp_max_dmg > max_dmg:
            max_dmg = current_opp_max_dmg
            primary_threat = opp
            max_type_mult = current_opp_max_mult

    opp_speed = calculate_effective_speed(primary_threat, state, 1 - unit_side)
    is_outsped = (unit_speed > opp_speed) if trick_room else (opp_speed > unit_speed)
        
    return primary_threat, max_dmg, is_outsped, max_type_mult

def estimate_incoming_threat(unit: BattlingPokemonView, unit_side: int, state: StateView, params: BattleRuleParam) -> Dict[str, Any]:
    opponents = state.sides[1 - unit_side].team.active
    threat_pkm, max_dmg, is_outsped, max_type_mult = _identify_biggest_threat_opponent(unit, unit_side, opponents, state, params)
    
    max_hp = unit.constants.stats[Stat.MAX_HP]
    
    threat_level = {
        "max_incoming_damage": max_dmg,
        "is_lethal": max_dmg >= unit.hp,
        "is_outsped": is_outsped,
        "penalty_multiplier": 1.0,
        "aggro_score": 0.0
    }

    if threat_level["is_lethal"] and threat_level["is_outsped"]:
        threat_level["penalty_multiplier"] = 0.01
        
    base_aggro = (max_dmg / max_hp) if max_hp > 0 else 0.0
    threat_level["aggro_score"] = base_aggro * max_type_mult
    
    return threat_level