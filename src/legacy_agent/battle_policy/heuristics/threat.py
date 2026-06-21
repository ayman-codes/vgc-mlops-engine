from typing import Any, Tuple, List, Optional
from poke_env.battle import Pokemon, AbstractBattle, MoveCategory, Weather

BOOST_MULTIPLIER = {
    -6: 2 / 8, -5: 2 / 7, -4: 2 / 6, -3: 2 / 5, -2: 2 / 4, -1: 2 / 3,
    0: 2 / 2, 1: 3 / 2, 2: 4 / 2, 3: 5 / 2, 4: 6 / 2, 5: 7 / 2, 6: 8 / 2
}

WEATHER_BOOSTS = {
    Weather.SUNNYDAY: {'fire': 1.5, 'water': 0.5},
    Weather.DESOLATELAND: {'fire': 1.5, 'water': 0.5},
    Weather.RAINDANCE: {'water': 1.5, 'fire': 0.5},
    Weather.PRIMORDIALSEA: {'water': 1.5, 'fire': 0.5},
}

def _identify_biggest_threat_opponent(
    unit: Pokemon,
    unit_side: int,
    opponents: List[Pokemon],
    state: AbstractBattle,
    params: Any
) -> Tuple[Optional[Pokemon], float, bool, float]:
    
    biggest_threat = None
    max_dmg = 0.0
    is_speed_threat = False
    speed_ratio = 1.0
    
    weather = state.weather if hasattr(state, 'weather') else {}
    weather_name = list(weather.keys())[0] if weather else None
    
    unit_speed = unit.base_stats.get('spe', 100)
    unit_boost = unit.boosts.get('spe', 0) if hasattr(unit, 'boosts') else 0
    effective_unit_speed = unit_speed * BOOST_MULTIPLIER.get(unit_boost, 1.0)
    
    for opp in opponents:
        if not opp or opp.fainted:
            continue
            
        opp_speed = opp.base_stats.get('spe', 100)
        opp_boost = opp.boosts.get('spe', 0) if hasattr(opp, 'boosts') else 0
        effective_opp_speed = opp_speed * BOOST_MULTIPLIER.get(opp_boost, 1.0)
        
        threat_score = 0.0
        for move in opp.moves.values():
            if not move.base_power:
                continue
                
            type_mult = unit.damage_multiplier(move)
            if type_mult == 0:
                continue
                
            opp_attack_stat = 'atk' if move.category == MoveCategory.PHYSICAL else 'spa'
            opp_stat_boost = opp.boosts.get(opp_attack_stat, 0) if hasattr(opp, 'boosts') else 0
            effective_attack = opp.base_stats.get(opp_attack_stat, 100) * BOOST_MULTIPLIER.get(opp_stat_boost, 1.0)
            
            unit_def_stat = 'def' if move.category == MoveCategory.PHYSICAL else 'spd'
            unit_def_boost = unit.boosts.get(unit_def_stat, 0) if hasattr(unit, 'boosts') else 0
            effective_defense = unit.base_stats.get(unit_def_stat, 100) * BOOST_MULTIPLIER.get(unit_def_boost, 1.0)
            
            base_dmg = (effective_attack / max(effective_defense, 1)) * move.base_power * type_mult
            
            weather_mod = 1.0
            if weather_name and weather_name in WEATHER_BOOSTS:
                move_type = getattr(move.type, 'name', str(move.type)).lower() if hasattr(move.type, 'name') else str(move.type).lower()
                weather_mod = WEATHER_BOOSTS[weather_name].get(move_type, 1.0)
                
            ability_mod = 1.0
            if opp.ability and opp.ability.lower() in ['adaptability', 'technician']:
                ability_mod = 1.5 if opp.ability.lower() == 'adaptability' else 1.5 if move.base_power <= 60 else 1.0
                
            move_threat = base_dmg * weather_mod * ability_mod
            if move_threat > threat_score:
                threat_score = move_threat
        
        speed_advantage = effective_opp_speed / max(effective_unit_speed, 1)
        if speed_advantage > 1.2:
            threat_score *= 1.2
            
        if threat_score > max_dmg:
            max_dmg = threat_score
            biggest_threat = opp
            is_speed_threat = speed_advantage > 1.2
            speed_ratio = speed_advantage
            
    return biggest_threat, float(max_dmg), is_speed_threat, speed_ratio
