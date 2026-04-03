from typing import List, Tuple
from vgc2.core.GameState import GameState
from vgc2.core.Pkm import Pkm

def calculate_effective_speed(pkm: Pkm, state: GameState) -> int:
    """
    Calculates the functional speed tier of a Pokémon considering field effects and modifiers.
    """
    base_speed = pkm.stats.spe
    stage = pkm.stat_stages.spe
    
    # Apply stat stage multipliers
    if stage >= 0:
        multiplier = (2 + stage) / 2
    else:
        multiplier = 2 / (2 - stage)
    
    speed = int(base_speed * multiplier)
    
    # Field effects 
    if state.field.tailwind[pkm.team_index]:
        speed *= 2
        
    # Trick Room inversion logic handled at comparison level 
    return speed

def _identify_biggest_threat_opponent(unit: Pkm, opponents: List[Pkm], state: GameState) -> Tuple[Pkm, float, bool]:
    """
    Identifies the opponent capable of dealing maximum damage to the unit.
    Returns: (Threatening Pkm, Max Damage, Is Outsped)
    """
    max_dmg = 0.0
    primary_threat = opponents[0]
    unit_speed = calculate_effective_speed(unit, state)
    trick_room = state.field.trick_room > 0

    for opp in opponents:
        if opp.hp <= 0:
            continue
            
            
        # Mapping resistances against predicted coverage 
        current_opp_max_dmg = 0.0
        for move in opp.moves:
            dmg = move.power # Placeholder for full damage formula
            if dmg > current_opp_max_dmg:
                current_opp_max_dmg = dmg
        
        if current_opp_max_dmg > max_dmg:
            max_dmg = current_opp_max_dmg
            primary_threat = opp

    opp_speed = calculate_effective_speed(primary_threat, state)
    
    # Turn-order resolution 
    if trick_room:
        is_outsped = unit_speed > opp_speed
    else:
        is_outsped = opp_speed > unit_speed
        
    return primary_threat, max_dmg, is_outsped

def estimate_incoming_threat(unit: Pkm, state: GameState) -> dict:
    """
    Evaluates survival probability and applies penalty multipliers based on speed tiers.
    """
    opponents = state.teams[1 - unit.team_index].active
    threat_pkm, max_dmg, is_outsped = _identify_biggest_threat_opponent(unit, opponents, state)
    
    threat_level = {
        "max_incoming_damage": max_dmg,
        "is_lethal": max_dmg >= unit.hp,
        "is_outsped": is_outsped,
        "penalty_multiplier": 1.0,
        "aggro_score": 0.0
    }

    # If opposing speed exceeds unit speed and damage exceeds HP, apply near-zero penalty
    if threat_level["is_lethal"] and threat_level["is_outsped"]:
        threat_level["penalty_multiplier"] = 0.01
        
    # Scale aggro-probability by type vulnerability
    # Placeholder for type-chart lookup
    threat_level["aggro_score"] = max_dmg / unit.max_hp
    
    return threat_level