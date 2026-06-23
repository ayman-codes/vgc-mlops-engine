"""Evolutionary Teambuilder using surrogate-assisted Genetic Algorithm.

Generates 6-Pokemon rosters by running the GA evolutionary loop, then
validating the top elite teams through empirical Showdown battles.
Integrates MLflow telemetry for fitness metrics and win rates.
"""

import asyncio

import mlflow

from poke_env.teambuilder import Teambuilder

from src.agent.selection_policy.double_oracle import Strategy
from src.agent.teambuild_policy.cache import (
    TeambuildCache,
    load_teambuild_cache,
)
from src.agent.teambuild_policy.evolution import run_evolution
from src.agent.teambuild_policy.battle_royale import run_battle_royale
from src.config.teambuild_config import TeambuildConfig


class EvolutionaryTeambuilder(Teambuilder):
    """Generates competitive 6-Pokemon rosters using a surrogate-assisted
    Genetic Algorithm followed by empirical battle validation.

    Phase 4: Runs the GA evolutionary loop with Bayesian MAP fitness
    to evolve a population of team chromosomes.
    Phase 5: Validates the top elite teams through cross-evaluation
    battles against baseline agents and selects the best team.

    Args:
        config: TeambuildConfig controlling GA parameters and fitness weight.
        cache: Pre-loaded TeambuildCache, or None to load from disk.
    """

    def __init__(
        self,
        config: TeambuildConfig | None = None,
        cache: TeambuildCache | None = None,
    ) -> None:
        self._config = config or TeambuildConfig()
        self._cache = cache

    def yield_team(self) -> str:
        """Execute the full teambuild pipeline and return the best team.

        Runs the evolutionary GA loop (Phase 4), then the empirical
        battle royale (Phase 5), and returns the single best team as
        a Showdown-formatted string.

        Returns:
            A valid Showdown team string with 6 Pokemon separated by
            double newlines. Returns empty string if the pipeline
            encounters an unrecoverable error.
        """
        cache = self._cache
        if cache is None:
            cache = load_teambuild_cache()

        try:
            with mlflow.start_run(nested=True, run_name="teambuild") as _run:
                mlflow.log_params({
                    "teambuild_population_size": self._config.population_size,
                    "teambuild_generations": self._config.generations,
                    "teambuild_mutation_rate": self._config.mutation_rate,
                    "teambuild_elite_fraction": self._config.elite_fraction,
                    "teambuild_fitness_archetype_weight": self._config.fitness_archetype_weight,
                })

                evo_results = run_evolution(self._config, cache)

                mlflow.log_metric(
                    "map_fitness_peak", evo_results["final_best_fitness"]
                )
                mlflow.log_metric(
                    "map_fitness_mean", evo_results["final_mean_fitness"]
                )
                mlflow.log_metric(
                    "generation_convergence_time", evo_results["convergence_time"]
                )

                best_teams = [
                    team.tolist() for team in evo_results["best_teams"]
                ]

                royale_results = asyncio.run(
                    run_battle_royale(
                        best_teams,
                        cache,
                        n_battles=self._config.battle_royale_n,
                    )
                )

                mlflow.log_metric(
                    "empirical_win_rate",
                    royale_results["empirical_win_rate"],
                )

                best_indices: list[int] = royale_results["best_team_indices"]
                team_pokemon = cache.hydrate_team(best_indices)
                strategy: Strategy = (0, 1, 2, 3, 4, 5)
                return cache.team_to_showdown(team_pokemon, strategy)

        except Exception:
            return ""
