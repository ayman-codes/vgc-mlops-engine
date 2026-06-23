"""Empirical validation engine for teambuild candidate teams.

Subjects the top elite teams from the surrogate-assisted Genetic Algorithm
to ground-truth Showdown battles against baseline agents. Selects the
team with the highest empirical win rate as the final output.
"""

from typing import Any

from poke_env import cross_evaluate
from poke_env.player import RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client import ServerConfiguration

from src.agent.teambuild_policy.cache import TeambuildCache


async def run_battle_royale(
    team_indices_list: list[list[int]],
    cache: TeambuildCache,
    n_battles: int = 100,
    server_url: str = "http://localhost:8000",
    battle_format: str = "gen9customgame",
) -> dict[str, Any]:
    """Run cross-evaluation battles to select the empirically best team.

    Hydrates each candidate team into a Showdown team string, creates
    SimpleHeuristicsPlayers, and pits them against baseline agents via
    cross_evaluate. The team with the highest overall win rate wins.

    Args:
        team_indices_list: List of candidate teams, each being a list of
            6 integer indices into cache.species_keys.
        cache: Initialized TeambuildCache for hydration and Showdown
            string generation.
        n_battles: Number of battles per player pair in cross_evaluate.
        server_url: URL of the local Showdown server.
        battle_format: Showdown battle format identifier.

    Returns:
        Dict with keys:
            best_team_idx: Index into team_indices_list of the winning team.
            best_team_indices: The winning team's species indices.
            empirical_win_rate: Win rate of the best team.
            all_win_rates: Dict mapping team index to win rate.
    """
    config = ServerConfiguration(server_url, server_url)
    showdown_teams: list[str] = []

    for indices in team_indices_list:
        team_pokemon = cache.hydrate_team(indices)
        strategy = (0, 1, 2, 3, 4, 5)
        team_str = cache.team_to_showdown(team_pokemon, strategy)
        showdown_teams.append(team_str)

    elite_players: list[SimpleHeuristicsPlayer] = []
    for i, team_str in enumerate(showdown_teams):
        player = SimpleHeuristicsPlayer(
            server_configuration=config,
            battle_format=battle_format,
            team=team_str,
            max_concurrent_battles=1,
        )
        elite_players.append(player)

    baseline_players = [
        RandomPlayer(
            server_configuration=config,
            battle_format=battle_format,
            max_concurrent_battles=1,
        ),
        MaxBasePowerPlayer(
            server_configuration=config,
            battle_format=battle_format,
            max_concurrent_battles=1,
        ),
        SimpleHeuristicsPlayer(
            server_configuration=config,
            battle_format=battle_format,
            max_concurrent_battles=1,
        ),
    ]

    all_players = elite_players + baseline_players
    n_elite = len(elite_players)

    await cross_evaluate(all_players, n_challenges=n_battles)

    win_rates: dict[int, float] = {}
    for i in range(n_elite):
        player = elite_players[i]
        if player.n_finished_battles > 0:
            win_rates[i] = (
                player.n_won_battles / player.n_finished_battles
            )
        else:
            win_rates[i] = 0.0

    if not win_rates:
        return {
            "best_team_idx": 0,
            "best_team_indices": team_indices_list[0],
            "empirical_win_rate": 0.0,
            "all_win_rates": {},
        }

    best_team_idx = max(win_rates, key=lambda k: win_rates[k])

    return {
        "best_team_idx": best_team_idx,
        "best_team_indices": team_indices_list[best_team_idx],
        "empirical_win_rate": win_rates[best_team_idx],
        "all_win_rates": win_rates,
    }
