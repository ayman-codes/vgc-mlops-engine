import itertools
from typing import List, Tuple, Dict

from vgc2.battle_engine import State, BattleEngine, BattlingTeam
from vgc2.battle_engine.team import Team
from vgc2.battle_engine.pokemon import Pokemon
from vgc2.battle_engine.view import PokemonView, TeamView, StateView
from vgc2.agent import BattlePolicy


def generate_team_combinations(source_team: Team, combination_size: int) -> List[Tuple[int, ...]]:
    if len(source_team.members) < combination_size:
        return []
    member_indices = list(range(len(source_team.members)))
    return list(itertools.combinations(member_indices, combination_size))


def run_sub_tournament(
    my_full_team: Team,
    my_pair_indices: Tuple[int, ...],
    opp_view_pair: Tuple[PokemonView, ...],
    predicted_builds_dict: Dict[PokemonView, List[Pokemon]],
    sim_policy: BattlePolicy
) -> float:
    opp_build_a = predicted_builds_dict.get(opp_view_pair[0], [])
    opp_build_b = predicted_builds_dict.get(opp_view_pair[1], [])
    
    if not opp_build_a or not opp_build_b:
        return 0.0
    
    sub_wins = 0
    sub_battles = 0
    build_matchups = itertools.product(opp_build_a, opp_build_b)
    
    for build_a, build_b in build_matchups:
        opp_predicted_pair = [build_a, build_b]
        my_pair_pokemon = [my_full_team.members[i] for i in my_pair_indices] 
        
        my_battling_team = BattlingTeam(active=my_pair_pokemon, reserve=[])
        opp_battling_team = BattlingTeam(active=opp_predicted_pair, reserve=[])
        
        initial_state = State((my_battling_team, opp_battling_team))
        engine = BattleEngine(initial_state)
        
        dummy_my_view = TeamView(my_full_team)
        dummy_opp_view = TeamView(Team(members=opp_predicted_pair))
        
        while not engine.finished():
            state_view_p0 = StateView(engine.state, 0, (dummy_my_view, dummy_opp_view))
            state_view_p1 = StateView(engine.state, 1, (dummy_opp_view, dummy_my_view))
            
            cmd_p0 = sim_policy.decision(state_view_p0)
            cmd_p1 = sim_policy.decision(state_view_p1)
            
            engine.run_turn((cmd_p0, cmd_p1))
            
        if engine.winning_side == 0:
            sub_wins += 1
        sub_battles += 1
        
    return float(sub_wins) / float(sub_battles) if sub_battles > 0 else 0.0