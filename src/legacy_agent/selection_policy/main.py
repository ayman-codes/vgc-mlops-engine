import itertools
from vgc2.agent import SelectionPolicy, SelectionCommand
from vgc2.battle_engine.team import Team
from vgc2.agent.battle import GreedyBattlePolicy
from src.agent.selection_policy.heuristics.matchup import generate_team_combinations, run_sub_tournament
from src.agent.selection_policy.heuristics.archetype import predict_opponent_builds
from src.config.selection_model import SelectionConfig

class MySelectionPolicy(SelectionPolicy): # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()
        self.sim_battle_policy = GreedyBattlePolicy()
        self.config = SelectionConfig()

    def decision(self, teams: tuple[Team, Team], max_size: int) -> SelectionCommand:
        my_full_team, opp_team_view = teams
        n_active = 2

        predicted_opponent_builds = {}
        all_opp_species_view = opp_team_view.members

        for opp_pkm_view in all_opp_species_view:
            predicted_builds = predict_opponent_builds(
                pokemon_view=opp_pkm_view,
                my_full_team=my_full_team,
                all_opp_views=all_opp_species_view,
                battle_params=self.sim_battle_policy.params,
                config = self.config
            )
            predicted_opponent_builds[opp_pkm_view] = predicted_builds
        
        my_potential_pairs = generate_team_combinations(my_full_team, n_active)
        opp_potential_pairs = list(itertools.combinations(all_opp_species_view, n_active))
        
        if not my_potential_pairs:
            return SelectionCommand(list(range(min(max_size, len(my_full_team.members)))))
        
        results = {pair: 0.0 for pair in my_potential_pairs}
        
        for my_pair in my_potential_pairs:
            total_win_rate = 0.0
            for opp_pair_view in opp_potential_pairs:
                win_rate = run_sub_tournament(
                    my_full_team=my_full_team,
                    my_pair_indices=my_pair,
                    opp_view_pair=opp_pair_view,
                    predicted_builds_dict=predicted_opponent_builds,
                    sim_policy=self.sim_battle_policy
                )
                total_win_rate += win_rate
            
            if opp_potential_pairs:
                results[my_pair] = total_win_rate / len(opp_potential_pairs)
        
        ranked_pairs = sorted(results.keys(), key=lambda p: results[p], reverse=True)
        num_pairs_to_select = max_size // n_active
        
        if len(ranked_pairs) < num_pairs_to_select:
            final_selection = []
            for pair in ranked_pairs:
                final_selection.extend(list(pair))
            remaining_indices = [i for i in range(len(my_full_team.members)) if i not in final_selection]
            final_selection.extend(remaining_indices)
            return SelectionCommand(final_selection[:max_size])
                
        final_selection = []
        for i in range(num_pairs_to_select):
            final_selection.extend(list(ranked_pairs[i]))

        return SelectionCommand(final_selection)