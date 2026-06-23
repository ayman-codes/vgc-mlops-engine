"""Evolutionary loop orchestrator for the Genetic Algorithm teambuild policy.

Executes the generational loop: Evaluate Fitness → Sort → Retain Elites →
Crossover → Mutate → Repeat. Returns the top elite teams and fitness
history for downstream validation and telemetry.
"""

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.config.teambuild_config import TeambuildConfig
from src.agent.teambuild_policy.cache import TeambuildCache
from src.agent.teambuild_policy.fitness import bayesian_map_fitness
from src.agent.teambuild_policy.operators import (
    init_population,
    crossover,
    mutate_team,
)


def run_evolution(
    config: TeambuildConfig,
    cache: TeambuildCache,
) -> dict[str, Any]:
    """Execute the full surrogate-assisted evolutionary loop.

    Iterates through config.generations, evaluating each team chromosome
    with Bayesian MAP fitness, retaining elite teams, and breeding new
    candidates via crossover and mutation.

    Args:
        config: TeambuildConfig controlling population size, generations,
            mutation rate, elite fraction, and fitness archetype weight.
        cache: Initialized TeambuildCache with usage data and GMM model.

    Returns:
        Dict with keys:
            best_teams: NDArray of shape (n_elites, 6) with top team indices.
            fitness_history: Dict with "best" and "mean" float lists per generation.
            convergence_time: Wall-clock seconds for the full loop.
            final_best_fitness: Peak fitness in the final generation.
            final_mean_fitness: Mean fitness in the final generation.
    """
    population = init_population(config.population_size, cache)
    n_elites = max(1, int(config.population_size * config.elite_fraction))
    fitness_history: dict[str, list[float]] = {"best": [], "mean": []}
    start_time = time.time()

    for _generation in range(config.generations):
        fitness_scores = np.array(
            [
                bayesian_map_fitness(
                    population[i].tolist(),
                    cache,
                    archetype_weight=config.fitness_archetype_weight,
                )
                for i in range(len(population))
            ],
            dtype=np.float64,
        )

        fitness_history["best"].append(float(fitness_scores.max()))
        fitness_history["mean"].append(float(fitness_scores.mean()))

        sorted_idx = np.argsort(-fitness_scores)
        population = population[sorted_idx]

        elites: NDArray[np.int64] = population[:n_elites].copy()

        new_pop: NDArray[np.int64] = elites.copy()
        while len(new_pop) < config.population_size:
            parent_a_idx = int(np.random.randint(0, len(population)))
            parent_b_idx = int(np.random.randint(0, len(population)))
            while parent_b_idx == parent_a_idx:
                parent_b_idx = int(np.random.randint(0, len(population)))

            child1, child2 = crossover(
                population[parent_a_idx], population[parent_b_idx], cache
            )
            child1 = mutate_team(child1, cache, config.mutation_rate)
            child2 = mutate_team(child2, cache, config.mutation_rate)

            new_pop = np.vstack([new_pop, child1.reshape(1, -1)])
            if len(new_pop) >= config.population_size:
                break
            new_pop = np.vstack([new_pop, child2.reshape(1, -1)])

        population = new_pop[: config.population_size]

    convergence_time = time.time() - start_time

    fitness_scores_final = np.array(
        [
            bayesian_map_fitness(
                population[i].tolist(),
                cache,
                archetype_weight=config.fitness_archetype_weight,
            )
            for i in range(len(population))
        ],
        dtype=np.float64,
    )
    sorted_final_idx = np.argsort(-fitness_scores_final)
    population = population[sorted_final_idx]
    best_teams = population[:n_elites]

    return {
        "best_teams": best_teams,
        "fitness_history": fitness_history,
        "convergence_time": convergence_time,
        "final_best_fitness": fitness_history["best"][-1],
        "final_mean_fitness": fitness_history["mean"][-1],
    }
