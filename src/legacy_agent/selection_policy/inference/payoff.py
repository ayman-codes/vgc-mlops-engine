import numpy as np
from typing import List, Tuple, Dict, Any

from vgc2.battle_engine.team import Team
from vgc2.battle_engine.view import PokemonView
from vgc2.battle_engine.pokemon import Pokemon
from vgc2.agent import BattlePolicy

from src.agent.selection_policy.heuristics.matchup import run_sub_tournament

def generate_payoff_matrix(
    my_full_team: Team,
    my_pairs: List[Tuple[int, ...]],
    opp_pairs: List[Tuple[PokemonView, ...]],
    predicted_builds_dict: Dict[PokemonView, List[Pokemon]],
    sim_policy: BattlePolicy
) -> np.ndarray[Any, Any]:
    """
    Generates a 2D payoff matrix for Nash Equilibrium resolution.
    Rows: Allied pair indices.
    Columns: Opponent pair views.
    Values: Expected win rates [0.0, 1.0].
    """
    num_rows = len(my_pairs)
    num_cols = len(opp_pairs)
    
    if num_rows == 0 or num_cols == 0:
        return np.zeros((0, 0), dtype=float)

    payoff_matrix = np.zeros((num_rows, num_cols), dtype=float)

    for i, my_pair in enumerate(my_pairs):
        for j, opp_pair in enumerate(opp_pairs):
            win_rate = run_sub_tournament(
                my_full_team=my_full_team,
                my_pair_indices=my_pair,
                opp_view_pair=opp_pair,
                predicted_builds_dict=predicted_builds_dict,
                sim_policy=sim_policy
            )
            payoff_matrix[i, j] = win_rate

    return payoff_matrix