from typing import List, Dict

from vgc2.battle_engine import State, BattlingTeam, calculate_damage
from vgc2.battle_engine.pokemon import Pokemon, PokemonSpecies, Move, BattlingPokemon
from vgc2.battle_engine.modifiers import Stat, Nature, Category
from vgc2.battle_engine.team import Team
from vgc2.battle_engine.view import PokemonView
from vgc2.battle_engine.constants import BattleRuleParam

from src.config.selection_model import SelectionConfig
from src.agent.selection_policy.heuristics.scoring import calculate_utility_score

def create_archetype_builds(species: PokemonSpecies, predicted_moveset: List[Move], config: SelectionConfig) -> List[Pokemon]:
    if not predicted_moveset:
        return []

    builds: List[Pokemon] = []
    base_stats = species.base_stats
    
    move_indices: List[int] = []
    for move in predicted_moveset:
        try:
            idx = species.moves.index(move)
            move_indices.append(idx)
        except ValueError:
            continue

    phys_moves = sum(1 for m in predicted_moveset if m.category == Category.PHYSICAL)
    spec_moves = sum(1 for m in predicted_moveset if m.category == Category.SPECIAL)
    
    is_physical_leaning = (base_stats[Stat.ATTACK] > base_stats[Stat.SPECIAL_ATTACK]) or \
                          (base_stats[Stat.ATTACK] == base_stats[Stat.SPECIAL_ATTACK] and phys_moves >= spec_moves)
    
    if is_physical_leaning:
        fast_nature = Nature.JOLLY
        fast_evs = (4, 252, 0, 0, 0, 252)
    else:
        fast_nature = Nature.TIMID
        fast_evs = (4, 0, 0, 252, 0, 252)
    builds.append(Pokemon(species=species, move_indexes=move_indices, nature=fast_nature, evs=fast_evs))

    if is_physical_leaning:
        bulky_nature = Nature.ADAMANT
        bulky_evs = (252, 252, 4, 0, 0, 0)
    else:
        bulky_nature = Nature.MODEST
        bulky_evs = (252, 0, 4, 252, 0, 0)
    builds.append(Pokemon(species=species, move_indexes=move_indices, nature=bulky_nature, evs=bulky_evs))

    is_physically_defensive = base_stats[Stat.DEFENSE] >= base_stats[Stat.SPECIAL_DEFENSE]
    if is_physically_defensive:
        def_nature = Nature.IMPISH if is_physical_leaning else Nature.BOLD
        def_evs = (252, 0, 252, 0, 4, 0)
    else:
        def_nature = Nature.CAREFUL if is_physical_leaning else Nature.CALM
        def_evs = (252, 0, 4, 0, 252, 0)
    builds.append(Pokemon(species=species, move_indexes=move_indices, nature=def_nature, evs=def_evs))
    
    atk_stat = float(base_stats[Stat.ATTACK])
    spa_stat = float(base_stats[Stat.SPECIAL_ATTACK])
    if abs(atk_stat - spa_stat) < config.heuristics.mixed_stat_threshold:
        mixed_nature = Nature.NAUGHTY if is_physical_leaning else Nature.RASH
        mixed_evs = (4, 252, 0, 252, 0, 0)
        builds.append(Pokemon(species=species, move_indexes=move_indices, nature=mixed_nature, evs=mixed_evs))

    return builds


def predict_moveset(
    attacker_species: PokemonSpecies, 
    my_full_team: Team, 
    all_opp_species_views: List[PokemonView], 
    battle_params: BattleRuleParam,
    config: SelectionConfig
) -> List[Move]:
    if not attacker_species.moves:
        return []

    move_scores: Dict[Move, float] = {move: 0.0 for move in attacker_species.moves}
    archetype_builds = create_archetype_builds(attacker_species, attacker_species.moves, config)
    
    if not archetype_builds:
        return []

    dummy_species = PokemonSpecies(base_stats=(1, 1, 1, 1, 1, 1), types=[], moves=[])
    dummy_pokemon = Pokemon(species=dummy_species, move_indexes=[])
    dummy_team = BattlingTeam(active=[dummy_pokemon], reserve=[])
    dummy_team.active = []
    my_battling_team = BattlingTeam(active=[p for p in my_full_team.members], reserve=[])
    neutral_state = State((my_battling_team, dummy_team))

    for move in attacker_species.moves:
        if move.base_power == 0 and move.category == Category.OTHER:
            move_scores[move] = calculate_utility_score(move, attacker_species, my_full_team, all_opp_species_views, battle_params, config)
            continue

        total_damage_across_builds = 0.0
        
        for attacker_build in archetype_builds:
            attacker_prototype = BattlingPokemon(attacker_build)
            total_damage_for_this_build = 0.0
            
            for my_pokemon in my_full_team.members:
                defender_prototype = BattlingPokemon(my_pokemon)
                damage = float(calculate_damage(
                    params=battle_params,
                    attacking_side=1,
                    move=move,
                    state=neutral_state,
                    attacker=attacker_prototype,
                    defender=defender_prototype
                ))
                max_hp = float(my_pokemon.stats[Stat.MAX_HP])
                if max_hp > 0.0:
                    total_damage_for_this_build += (damage / max_hp) * 100.0
            
            team_size = len(my_full_team.members)
            avg_damage_for_build = total_damage_for_this_build / team_size if team_size > 0 else 0.0
            total_damage_across_builds += avg_damage_for_build

        num_builds = len(archetype_builds)
        move_scores[move] = total_damage_across_builds / num_builds if num_builds > 0 else 0.0

    sorted_moves = sorted(move_scores.keys(), key=lambda m: move_scores[m], reverse=True)
    return sorted_moves[:4]


def predict_opponent_builds(
    pokemon_view: PokemonView, 
    my_full_team: Team, 
    all_opp_views: List[PokemonView], 
    battle_params: BattleRuleParam,
    config: SelectionConfig
) -> List[Pokemon]:
    species = pokemon_view.species
    if not species:
        return []
    predicted_moveset = predict_moveset(species, my_full_team, all_opp_views, battle_params, config)
    return create_archetype_builds(species, predicted_moveset, config)