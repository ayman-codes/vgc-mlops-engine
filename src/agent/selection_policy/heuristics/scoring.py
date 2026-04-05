from typing import List, Union
from vgc2.battle_engine import State, BattlingTeam, calculate_damage
from vgc2.battle_engine.pokemon import Pokemon, PokemonSpecies, Move, BattlingPokemon
from vgc2.battle_engine.modifiers import Stat, Type, Category, Status, Weather, Terrain
from vgc2.battle_engine.team import Team
from vgc2.battle_engine.view import PokemonView
from vgc2.battle_engine.constants import BattleRuleParam
from src.config.selection_model import SelectionConfig

def _get_field_effect_damage_swing(
    move_type_boost: Type,
    damage_multiplier: float,
    team_to_evaluate: Union[List[Pokemon], List[PokemonView]],
    battle_params: BattleRuleParam
) -> float:
    net_damage_gain = 0.0
    for pkm in team_to_evaluate:
        best_move = None
        max_power = -1.0
        potential_moves = pkm.moves if hasattr(pkm, 'moves') else pkm.species.moves
        
        for pkm_move in potential_moves:
            if pkm_move.pkm_type == move_type_boost and pkm_move.base_power > max_power:
                max_power = float(pkm_move.base_power)
                best_move = pkm_move
        
        if best_move:
            generic_defender_species = PokemonSpecies(base_stats=(80, 80, 80, 80, 80, 80), types=[], moves=[])
            generic_defender = BattlingPokemon(Pokemon(species=generic_defender_species, move_indexes=[]))
            
            attacker_proto = BattlingPokemon(pkm) if isinstance(pkm, Pokemon) else BattlingPokemon(Pokemon(species=pkm.species, move_indexes=[]))
            
            state_no_effect = State((BattlingTeam([generic_defender], []), BattlingTeam([attacker_proto], [])))
            base_damage = calculate_damage(params=battle_params, attacking_side=1, move=best_move, state=state_no_effect, attacker=attacker_proto, defender=generic_defender)
            
            boosted_damage = float(base_damage) * damage_multiplier
            net_damage_gain += (boosted_damage - float(base_damage))
            
    return net_damage_gain

def calculate_utility_score(
    move: Move,
    attacker_species: PokemonSpecies,
    my_full_team: Team,
    all_opp_species_views: List[PokemonView],
    battle_params: BattleRuleParam,
    config: SelectionConfig
) -> float:
    score = 0.0
    my_team_members = my_full_team.members
    heuristics = config.heuristics

    if move.protect:
        max_damage = 0.0
        defender_proto = BattlingPokemon(Pokemon(species=attacker_species, move_indexes=[]))
        neutral_state_pkm = BattlingTeam([defender_proto], [])

        for my_pkm in my_full_team.members:
            for my_move in my_pkm.moves:
                if my_move.base_power > 0:
                    attacker_proto = BattlingPokemon(my_pkm)
                    temp_state = State((BattlingTeam([attacker_proto], []), neutral_state_pkm))
                    
                    damage = float(calculate_damage(params=battle_params, attacking_side=0, move=my_move, state=temp_state, attacker=attacker_proto, defender=defender_proto))
                    if damage > max_damage:
                        max_damage = damage
        
        attacker_hp = float(attacker_species.base_stats[Stat.MAX_HP])
        return (max_damage / attacker_hp) * 100.0 if attacker_hp > 0 else 0.0

    if move.status == Status.BURN:
        score += (1.0 / heuristics.base_hp_reference_turns) * 100.0 * heuristics.status_score_weight
        my_phys_attackers = [p for p in my_full_team.members if p.stats[Stat.ATTACK] > p.stats[Stat.SPECIAL_ATTACK]]
        if my_phys_attackers:
            strongest_phys_attacker = max(my_phys_attackers, key=lambda p: p.stats[Stat.ATTACK])
            best_phys_move = max(strongest_phys_attacker.moves, key=lambda m: m.base_power if m.category == Category.PHYSICAL else -1.0)

            if best_phys_move.base_power > 0:
                attacker_proto = BattlingPokemon(strongest_phys_attacker)
                defender_proto = BattlingPokemon(Pokemon(species=attacker_species, move_indexes=[]))
                temp_state = State((BattlingTeam([attacker_proto], []), BattlingTeam([defender_proto], [])))
                
                dmg_unburned = float(calculate_damage(params=battle_params, attacking_side=0, move=best_phys_move, state=temp_state, attacker=attacker_proto, defender=defender_proto))
                
                attacker_proto.status = Status.BURN
                dmg_burned = float(calculate_damage(params=battle_params, attacking_side=0, move=best_phys_move, state=temp_state, attacker=attacker_proto, defender=defender_proto))
                
                damage_prevented = dmg_unburned - dmg_burned
                defender_hp = float(attacker_species.base_stats[Stat.MAX_HP])
                score += ((damage_prevented / defender_hp) * 100.0) if defender_hp > 0 else 0.0

    elif move.status == Status.TOXIC:
        score += heuristics.toxic_damage_coefficient * 100.0 * heuristics.status_score_weight

    elif move.status == Status.PARALYZED:
        if my_full_team.members:
            fastest_pkm = max(my_full_team.members, key=lambda p: p.stats[Stat.SPEED])
            if fastest_pkm.moves:
                best_move = max(fastest_pkm.moves, key=lambda m: m.base_power)
                if best_move.base_power > 0:
                    attacker_proto = BattlingPokemon(fastest_pkm)
                    defender_proto = BattlingPokemon(Pokemon(species=attacker_species, move_indexes=[]))
                    temp_state = State((BattlingTeam([attacker_proto], []), BattlingTeam([defender_proto], [])))

                    damage_potential = float(calculate_damage(params=battle_params, attacking_side=0, move=best_move, state=temp_state, attacker=attacker_proto, defender=defender_proto))
                    damage_denied = damage_potential * heuristics.paralysis_denial_chance
                    defender_hp = float(attacker_species.base_stats[Stat.MAX_HP])
                    score += ((damage_denied / defender_hp) * 100.0) if defender_hp > 0 else 0.0

    elif move.status == Status.SLEEP:
        max_damage_potential = 0.0
        all_move_indices = list(range(len(attacker_species.moves)))
        attacker_proto = BattlingPokemon(Pokemon(species=attacker_species, move_indexes=all_move_indices))
        
        for my_pkm in my_full_team.members:
            defender_proto = BattlingPokemon(my_pkm)
            temp_state = State((BattlingTeam([defender_proto], []), BattlingTeam([attacker_proto], [])))
            for opp_move in attacker_species.moves:
                if opp_move.base_power > 0:
                    damage = float(calculate_damage(params=battle_params, attacking_side=1, move=opp_move, state=temp_state, attacker=attacker_proto, defender=defender_proto))
                    if damage > max_damage_potential:
                        max_damage_potential = damage
        
        avg_my_hp = float(sum(p.stats[Stat.MAX_HP] for p in my_full_team.members) / len(my_full_team.members)) if my_full_team.members else 1.0
        score += ((max_damage_potential * heuristics.sleep_denial_turns) / avg_my_hp * 100.0) if avg_my_hp > 0 else 0.0

    if move.weather_start == Weather.RAIN:
        opp_gain = _get_field_effect_damage_swing(Type.WATER, 1.5, all_opp_species_views, battle_params)
        my_gain_from_nerf = _get_field_effect_damage_swing(Type.FIRE, 0.5, all_opp_species_views, battle_params)
        score += (opp_gain - my_gain_from_nerf) * heuristics.weather_score_weight

    elif move.weather_start == Weather.SUN:
        opp_gain = _get_field_effect_damage_swing(Type.FIRE, 1.5, all_opp_species_views, battle_params)
        my_gain_from_nerf = _get_field_effect_damage_swing(Type.WATER, 0.5, all_opp_species_views, battle_params)
        score += (opp_gain - my_gain_from_nerf) * heuristics.weather_score_weight

    elif move.weather_start == Weather.SAND:
        non_immune_me = sum(1 for p in my_team_members if not any(t in p.species.types for t in [Type.ROCK, Type.GROUND, Type.STEEL]))
        base_hp = float(my_team_members[0].stats[Stat.MAX_HP]) if my_team_members else 1.0
        score += (float(non_immune_me) * (base_hp / heuristics.base_hp_reference_turns)) * heuristics.weather_score_weight
        opp_rock_types = sum(1 for p in all_opp_species_views if Type.ROCK in p.species.types)
        score += (float(opp_rock_types) * 20.0) * heuristics.weather_score_weight

    elif move.weather_start == Weather.SNOW:
        non_immune_me = sum(1 for p in my_team_members if Type.ICE not in p.species.types)
        base_hp = float(my_team_members[0].stats[Stat.MAX_HP]) if my_team_members else 1.0
        score += (float(non_immune_me) * (base_hp / heuristics.base_hp_reference_turns)) * heuristics.weather_score_weight
        opp_ice_types = sum(1 for p in all_opp_species_views if Type.ICE in p.species.types)
        score += (float(opp_ice_types) * 20.0) * heuristics.weather_score_weight

    standard_damage_unit = (float(my_team_members[0].stats[Stat.MAX_HP]) / heuristics.base_hp_reference_turns) if my_team_members else 0.0

    if move.field_start == Terrain.ELECTRIC_TERRAIN:
        opp_gain = _get_field_effect_damage_swing(Type.ELECTRIC, 1.3, all_opp_species_views, battle_params)
        score += opp_gain * heuristics.terrain_score_weight
        i_have_sleep_moves = any(any(m.status == Status.SLEEP for m in p.moves) for p in my_team_members)
        if i_have_sleep_moves:
            avg_hp = float(sum(p.stats[Stat.MAX_HP] for p in my_team_members) / len(my_team_members)) if my_team_members else 1.0
            best_sleep_bonus = 0.0
            for p in my_team_members:
                for m in p.moves:
                    if m.status == Status.SLEEP:
                        potential = ((float(m.base_power) * float(p.stats[Stat.ATTACK])) / avg_hp) * heuristics.sleep_denial_turns * 100.0
                        if potential > best_sleep_bonus:
                            best_sleep_bonus = potential
            score += best_sleep_bonus * heuristics.terrain_score_weight

    elif move.field_start == Terrain.GRASSY_TERRAIN:
        is_opp_grass = sum(1 for p in all_opp_species_views if Type.GRASS in p.species.types)
        is_opp_flying = sum(1 for p in all_opp_species_views if Type.FLYING in p.species.types)
        score += (float(is_opp_grass) * (standard_damage_unit * 0.75)) * heuristics.terrain_score_weight
        score += (float(6 - is_opp_flying) * (standard_damage_unit / 2.0)) * heuristics.terrain_score_weight

    elif move.field_start == Terrain.PSYCHIC_TERRAIN:
        is_opp_psychic = sum(1 for p in all_opp_species_views if Type.PSYCHIC in p.species.types)
        is_opp_flying = sum(1 for p in all_opp_species_views if Type.FLYING in p.species.types)
        score += (float(is_opp_psychic) * (standard_damage_unit * 0.75)) * heuristics.terrain_score_weight
        can_i_use_priority = any(any(m.priority > 0 for m in p.moves) for p in my_team_members)
        if can_i_use_priority:
            score += (float(6 - is_opp_flying) * 15.0) * heuristics.terrain_score_weight

    elif move.field_start == Terrain.MISTY_TERRAIN:
        is_me_dragon = sum(1 for p in my_team_members if Type.DRAGON in p.species.types)
        score += (float(is_me_dragon) * (standard_damage_unit / 2.0)) * heuristics.terrain_score_weight
        can_i_use_status = any(any(m.status in {Status.SLEEP, Status.BURN, Status.TOXIC, Status.PARALYZED} for m in p.moves) for p in my_team_members)
        is_opp_flying = sum(1 for p in all_opp_species_views if Type.FLYING in p.species.types)
        if can_i_use_status:
            score += (float(6 - is_opp_flying) * 15.0) * heuristics.terrain_score_weight

    return score