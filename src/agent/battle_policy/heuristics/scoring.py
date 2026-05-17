from typing import Any
from poke_env.battle import Pokemon, Move, AbstractBattle, MoveCategory, Weather

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

def _get_stat_with_boosts(pokemon: Pokemon, stat: str) -> float:
    base = pokemon.base_stats.get(stat, 100)
    boost = pokemon.boosts.get(stat, 0) if hasattr(pokemon, 'boosts') else 0
    multiplier = BOOST_MULTIPLIER.get(boost, 1.0)
    return float(base * multiplier)

def _score_single_offensive_move(unit: Pokemon, opp: Pokemon, move: Move, state: AbstractBattle, params: Any, target_idx: int) -> float:
    if not move.base_power:
        return 0.0

    type_multiplier = opp.damage_multiplier(move)
    if type_multiplier == 0:
        return -100.0

    stab = 1.5 if move.type in unit.types else 1.0
    effective_stat = _get_stat_with_boosts(unit, 'atk' if move.category == MoveCategory.PHYSICAL else 'spa')
    opp_defense = _get_stat_with_boosts(opp, 'def' if move.category == MoveCategory.PHYSICAL else 'spd')

    base_damage = (effective_stat / max(opp_defense, 1)) * move.base_power * type_multiplier * stab
    accuracy_factor = move.accuracy / 100.0 if move.accuracy else 1.0
    expected_hits = getattr(move, 'expected_hits', 1) or 1

    weather = state.weather if hasattr(state, 'weather') else {}
    weather_name = list(weather.keys())[0] if weather else None
    weather_mod = 1.0
    if weather_name and weather_name in WEATHER_BOOSTS:
        move_type = getattr(move.type, 'name', str(move.type)).lower() if hasattr(move.type, 'name') else str(move.type).lower()
        weather_mod = WEATHER_BOOSTS[weather_name].get(move_type, 1.0)

    pp_penalty = move.current_pp / move.max_pp if hasattr(move, 'max_pp') and move.max_pp > 0 else 1.0

    score = base_damage * accuracy_factor * expected_hits * weather_mod * (0.8 + 0.2 * pp_penalty)
    return float(score)

def _score_protect_move(unit: Pokemon, slot_idx: int, state: AbstractBattle, params: Any) -> float:
    hp_fraction = unit.current_hp_fraction if hasattr(unit, 'current_hp_fraction') else 1.0
    base_score = 10.0 * (1.0 - hp_fraction * 0.5)

    opponents = state.opponent_active_pokemon if hasattr(state, 'opponent_active_pokemon') else []
    max_threat = 0.0
    for opp in opponents:
        if opp and not opp.fainted:
            for m in opp.moves.values():
                if m.base_power:
                    threat = m.base_power * unit.damage_multiplier(m)
                    max_threat = max(max_threat, threat)

    threat_multiplier = min(max_threat / 100.0, 2.0)
    return float(base_score * (1.0 + threat_multiplier * 0.5))

def _score_single_switch_action(unit: Pokemon, reserve: Pokemon, slot_idx: int, state: AbstractBattle, params: Any) -> float:
    opponents = state.opponent_active_pokemon if hasattr(state, 'opponent_active_pokemon') else []
    unit_resistance = 0.0
    reserve_resistance = 0.0

    for opp in opponents:
        if opp and not opp.fainted:
            for m in opp.moves.values():
                if m.base_power:
                    unit_resistance += unit.damage_multiplier(m)
                    reserve_resistance += reserve.damage_multiplier(m)

    if unit_resistance == 0:
        unit_resistance = 1.0

    defensive_improvement = (unit_resistance - reserve_resistance) / unit_resistance
    stat_improvement = (sum(reserve.base_stats.values()) - sum(unit.base_stats.values())) / max(sum(unit.base_stats.values()), 1)

    return float(5.0 + defensive_improvement * 10.0 + stat_improvement * 5.0)
