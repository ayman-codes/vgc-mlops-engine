from typing import List, Tuple, Optional
from vgc2.battle_engine.view import StateView, BattlingPokemonView
from vgc2.battle_engine.modifiers import Stat

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

def _identify_biggest_threat_opponent(unit: BattlingPokemonView, unit_side: int, opponents: List[BattlingPokemonView], state: StateView) -> Tuple[Optional[BattlingPokemonView], float, bool]:
    max_dmg = 0.0
    primary_threat = None
    
    valid_opponents = [opp for opp in opponents if opp and opp.hp > 0]
    if not valid_opponents:
        return None, 0.0, False

    primary_threat = valid_opponents[0]
    unit_speed = calculate_effective_speed(unit, state, unit_side)
    trick_room = state.trickroom

    for opp in valid_opponents:
        current_opp_max_dmg = 0.0
        if opp.battling_moves:
            for move in opp.battling_moves:
                dmg = float(move.constants.base_power) 
                if dmg > current_opp_max_dmg:
                    current_opp_max_dmg = dmg
        
        if current_opp_max_dmg > max_dmg:
            max_dmg = current_opp_max_dmg
            primary_threat = opp

    opp_speed = calculate_effective_speed(primary_threat, state, 1 - unit_side)
    
    if trick_room:
        is_outsped = unit_speed > opp_speed
    else:
        is_outsped = opp_speed > unit_speed
        
    return primary_threat, max_dmg, is_outsped

def estimate_incoming_threat(unit: BattlingPokemonView, unit_side: int, state: StateView) -> dict:
    opponents = state.sides[1 - unit_side].team.active
    threat_pkm, max_dmg, is_outsped = _identify_biggest_threat_opponent(unit, unit_side, opponents, state)
    
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
        
    threat_level["aggro_score"] = (max_dmg / max_hp) if max_hp > 0 else 0.0
    
    return threat_level