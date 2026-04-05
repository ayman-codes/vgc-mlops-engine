from __future__ import annotations
from typing import List
from vgc2.battle_engine import State, BattlingTeam, calculate_damage
from vgc2.battle_engine.pokemon import Pokemon, PokemonSpecies, Move, BattlingPokemon
from vgc2.battle_engine.modifiers import Stat, Type, Category, Status, Weather, Terrain
from vgc2.battle_engine.team import Team
from vgc2.battle_engine.view import PokemonView
from vgc2.battle_engine.constants import BattleRuleParam
from src.config.selection_model import SelectionHeuristicsConfig
from src.agent.selection_policy.heuristics.archetype import create_archetype_builds


def _calculate_protect_score(
    move: Move,
    attacker_species: PokemonSpecies,
    my_full_team: Team,
    params: BattleRuleParam,
) -> float:
    max_damage = 0
    defender_proto = BattlingPokemon(Pokemon(species=attacker_species, move_indexes=[]))
    neutral_def_team = BattlingTeam(active=[defender_proto], reserve=[])

    for my_pkm in my_full_team.members:
        for my_move in my_pkm.moves:
            if my_move.base_power > 0:
                attacker_proto = BattlingPokemon(my_pkm)
                temp_state = State((BattlingTeam([attacker_proto], []), neutral_def_team))
                dmg = calculate_damage(params, 0, my_move, temp_state, attacker_proto, defender_proto)
                if dmg > max_damage:
                    max_damage = dmg

    hp = attacker_species.base_stats[Stat.MAX_HP]
    return (max_damage / hp) * 100 if hp > 0 else 0.0


def _calculate_status_score(
    move: Move,
    attacker_species: PokemonSpecies,
    my_full_team: Team,
    all_opp_views: List[PokemonView],
    params: BattleRuleParam,
    config: SelectionHeuristicsConfig,
) -> float:
    score = 0.0
    if move.status == Status.BURN:
        score += (1 / 16) * 100
        phys_attackers = [p for p in my_full_team.members if p.stats[Stat.ATTACK] > p.stats[Stat.SPECIAL_ATTACK]]
        if phys_attackers:
            strongest = max(phys_attackers, key=lambda p: p.stats[Stat.ATTACK])
            best_move = max(strongest.moves, key=lambda m: m.base_power if m.category == Category.PHYSICAL else -1)
            if best_move.base_power > 0:
                att = BattlingPokemon(strongest)
                defn = BattlingPokemon(Pokemon(species=attacker_species, move_indexes=[]))
                st = State((BattlingTeam([att], []), BattlingTeam([defn], [])))
                unburned = calculate_damage(params, 0, best_move, st, att, defn)
                att.status = Status.BURN
                burned = calculate_damage(params, 0, best_move, st, att, defn)
                prevented = unburned - burned
                hp = attacker_species.base_stats[Stat.MAX_HP]
                score += (prevented / hp) * 100 if hp > 0 else 0
    elif move.status == Status.TOXIC:
        score += config.toxic_damage_coefficient * 100
    elif move.status == Status.PARALYZED:
        fastest = max(my_full_team.members, key=lambda p: p.stats[Stat.SPEED])
        best_move = max(fastest.moves, key=lambda m: m.base_power)
        if best_move.base_power > 0:
            att = BattlingPokemon(fastest)
            defn = BattlingPokemon(Pokemon(species=attacker_species, move_indexes=[]))
            st = State((BattlingTeam([att], []), BattlingTeam([defn], [])))
            potential = calculate_damage(params, 0, best_move, st, att, defn)
            hp = attacker_species.base_stats[Stat.MAX_HP]
            score += ((potential * config.paralysis_denial_chance) / hp) * 100 if hp > 0 else 0
    elif move.status == Status.SLEEP:
        max_potential = 0
        all_moves = list(range(len(attacker_species.moves)))
        att = BattlingPokemon(Pokemon(species=attacker_species, move_indexes=all_moves))
        for my_pkm in my_full_team.members:
            defn = BattlingPokemon(my_pkm)
            st = State((BattlingTeam([defn], []), BattlingTeam([att], [])))
            for opp_move in attacker_species.moves:
                if opp_move.base_power > 0:
                    dmg = calculate_damage(params, 1, opp_move, st, att, defn)
                    if dmg > max_potential:
                        max_potential = dmg
        avg_hp = sum(p.stats[Stat.MAX_HP] for p in my_full_team.members) / len(my_full_team.members)
        score += (max_potential * config.sleep_denial_turns / avg_hp) * 100 if avg_hp > 0 else 0
    return score * config.status_score_weight


def _calculate_field_effect_score(
    move: Move,
    my_full_team: Team,
    all_opp_views: List[PokemonView],
    params: BattleRuleParam,
    config: SelectionHeuristicsConfig,
) -> float:
    score = 0.0

    def get_field_swing(boost_type: Type, mult: float, targets: list) -> float:
        gain = 0.0
        for p in targets:
            moves = p.moves if hasattr(p, "moves") else p.species.moves
            best = max((m for m in moves if m.pkm_type == boost_type and m.base_power > 0), key=lambda m: m.base_power, default=None)
            if best:
                defn = PokemonSpecies(base_stats=(80, 80, 80, 80, 80, 80), types=[], moves=[])
                def_p = BattlingPokemon(Pokemon(species=defn, move_indexes=[]))
                att_p = BattlingPokemon(p) if hasattr(p, "moves") else BattlingPokemon(Pokemon(species=p.species, move_indexes=[]))
                base = calculate_damage(params, 1, best, State((BattlingTeam([def_p], []), BattlingTeam([att_p], []))), att_p, def_p)
                gain += (base * mult) - base
        return gain

    if move.weather_start == Weather.RAIN:
        score += get_field_swing(Type.WATER, 1.5, all_opp_views) - get_field_swing(Type.FIRE, 0.5, all_opp_views)
    elif move.weather_start == Weather.SUN:
        score += get_field_swing(Type.FIRE, 1.5, all_opp_views) - get_field_swing(Type.WATER, 0.5, all_opp_views)
    elif move.weather_start == Weather.SAND:
        non_immune = sum(1 for p in my_full_team.members if not any(t in p.species.types for t in [Type.ROCK, Type.GROUND, Type.STEEL]))
        score += non_immune * (my_full_team.members[0].stats[Stat.MAX_HP] / 16) + sum(1 for p in all_opp_views if Type.ROCK in p.species.types) * 20
    elif move.weather_start == Weather.SNOW:
        non_immune = sum(1 for p in my_full_team.members if Type.ICE not in p.species.types)
        score += non_immune * (my_full_team.members[0].stats[Stat.MAX_HP] / 16) + sum(1 for p in all_opp_views if Type.ICE in p.species.types) * 20
    elif move.field_start == Terrain.ELECTRIC_TERRAIN:
        score += get_field_swing(Type.ELECTRIC, 1.3, all_opp_views)
        if any(any(m.status == Status.SLEEP for m in p.moves) for p in my_full_team.members):
            avg_hp = sum(p.stats[Stat.MAX_HP] for p in my_full_team.members) / len(my_full_team.members)
            bonus = max((m.base_power * p.stats[Stat.ATTACK] / avg_hp * 1.5 * 100 for p in my_full_team.members for m in p.moves if m.status == Status.SLEEP), default=0)
            score += bonus
    elif move.field_start == Terrain.GRASSY_TERRAIN:
        std = my_full_team.members[0].stats[Stat.MAX_HP] / 16
        score += sum(1 for p in all_opp_views if Type.GRASS in p.species.types) * (std * 0.75) + (6 - sum(1 for p in all_opp_views if Type.FLYING in p.species.types)) * (std / 2)
    elif move.field_start == Terrain.PSYCHIC_TERRAIN:
        std = my_full_team.members[0].stats[Stat.MAX_HP] / 16
        score += sum(1 for p in all_opp_views if Type.PSYCHIC in p.species.types) * (std * 0.75)
        if any(any(m.priority > 0 for m in p.moves) for p in my_full_team.members):
            score += (6 - sum(1 for p in all_opp_views if Type.FLYING in p.species.types)) * 15
    elif move.field_start == Terrain.MISTY_TERRAIN:
        std = my_full_team.members[0].stats[Stat.MAX_HP] / 16
        score += sum(1 for p in my_full_team.members if Type.DRAGON in p.species.types) * (std / 2)
        if any(any(m.status in {Status.SLEEP, Status.BURN, Status.TOXIC, Status.PARALYZED} for m in p.moves) for p in my_full_team.members):
            score += (6 - sum(1 for p in all_opp_views if Type.FLYING in p.species.types)) * 15
    return score * config.terrain_score_weight


def calculate_utility_score(
    move: Move,
    attacker_species: PokemonSpecies,
    my_full_team: Team,
    all_opp_views: List[PokemonView],
    params: BattleRuleParam,
    config: SelectionHeuristicsConfig,
) -> float:
    if move.base_power == 0 and move.category == Category.OTHER:
        if move.protect:
            return _calculate_protect_score(move, attacker_species, my_full_team, params)
        if move.status:
            return _calculate_status_score(move, attacker_species, my_full_team, all_opp_views, params, config)
        if move.weather_start or move.field_start:
            return _calculate_field_effect_score(move, my_full_team, all_opp_views, params, config)
    return 0.0


def calculate_damage_score(
    move: Move,
    attacker_species: PokemonSpecies,
    my_full_team: Team,
    params: BattleRuleParam,
    config: SelectionHeuristicsConfig,
) -> float:
    if move.base_power == 0 or move.category not in (Category.PHYSICAL, Category.SPECIAL):
        return 0.0

    builds = create_archetype_builds(attacker_species, [move], config)
    if not builds:
        return 0.0

    neutral_species = PokemonSpecies(base_stats=(1, 1, 1, 1, 1, 1), types=[], moves=[])
    dummy_team = BattlingTeam(active=[], reserve=[])
    my_team = BattlingTeam(active=[p for p in my_full_team.members], reserve=[])
    neutral_state = State((my_team, dummy_team))

    total_damage = 0.0
    for build in builds:
        attacker = BattlingPokemon(build)
        dmg_sum = sum(
            calculate_damage(params, 1, move, neutral_state, attacker, BattlingPokemon(my_p)) / my_p.stats[Stat.MAX_HP] * 100
            for my_p in my_full_team.members if my_p.stats[Stat.MAX_HP] > 0
        )
        total_damage += dmg_sum / len(my_full_team.members) if my_full_team.members else 0

    return total_damage / len(builds) if builds else 0.0


def score_move(
    move: Move,
    attacker_species: PokemonSpecies,
    my_full_team: Team,
    all_opp_views: List[PokemonView],
    params: BattleRuleParam,
    config: SelectionHeuristicsConfig,
) -> float:
    if move.base_power == 0 and move.category == Category.OTHER:
        return calculate_utility_score(move, attacker_species, my_full_team, all_opp_views, params, config)
    return calculate_damage_score(move, attacker_species, my_full_team, params, config)