"""Domain-aware genetic operators for team chromosome evolution.

Operates exclusively on integer arrays of shape (population_size, 6)
where each row is a team chromosome of 6 distinct species indices
referencing cache.species_keys. All operators enforce the hard
constraint of no duplicate species within a single team.
"""

import numpy as np
from numpy.typing import NDArray

from src.agent.teambuild_policy.cache import TeambuildCache


def _sample_unique_indices(
    cache: TeambuildCache,
    n: int = 6,
) -> NDArray[np.int64]:
    """Sample n unique species indices weighted by usage without replacement.

    Uses rejection sampling: draws from the usage-weighted distribution,
    skipping indices already selected.

    Args:
        cache: Initialized TeambuildCache with usage_weights.
        n: Number of unique indices to sample.

    Returns:
        Array of n unique int64 indices into cache.species_keys.

    Raises:
        ValueError: If n_species is less than n (insufficient pool).
    """
    if cache.n_species < n:
        raise ValueError(
            f"Species pool size {cache.n_species} is too small "
            f"to sample {n} unique indices"
        )
    indices = np.arange(cache.n_species, dtype=np.int64)
    probs = np.asarray(cache.usage_weights, dtype=np.float64).copy()
    selected: list[int] = []
    while len(selected) < n:
        remaining_mask = np.ones(cache.n_species, dtype=bool)
        for s in selected:
            remaining_mask[s] = False
        if not remaining_mask.any():
            break
        remaining_probs = probs * remaining_mask.astype(np.float64)
        total = remaining_probs.sum()
        if total <= 0:
            remaining = indices[remaining_mask]
            selected.append(int(np.random.choice(remaining)))
            continue
        remaining_probs /= total
        pick = int(np.random.choice(indices, p=remaining_probs))
        selected.append(pick)
    return np.array(selected[:n], dtype=np.int64)


def init_population(
    pop_size: int,
    cache: TeambuildCache,
) -> NDArray[np.int64]:
    """Generate an initial population of team chromosomes.

    Each team is an array of 6 unique species indices sampled from the
    usage-weighted distribution. No duplicate species appear within a
    single team.

    Args:
        pop_size: Number of teams in the population.
        cache: Initialized TeambuildCache with usage_weights.

    Returns:
        Int64 array of shape (pop_size, 6) where each row is a valid
        team chromosome.

    Raises:
        ValueError: If cache has fewer than 6 species.
    """
    if cache.n_species < 6:
        raise ValueError(
            f"Need at least 6 species in cache, got {cache.n_species}"
        )
    population = np.empty((pop_size, 6), dtype=np.int64)
    for i in range(pop_size):
        population[i] = _sample_unique_indices(cache, n=6)
    return population


def crossover(
    parent1: NDArray[np.int64],
    parent2: NDArray[np.int64],
    cache: TeambuildCache,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Perform single-point crossover between two parent chromosomes.

    Applies single-point crossover at a random index 1-5, then replaces
    any duplicate species in each child by sampling new unique indices
    from unused species. Both resulting children are guaranteed to have
    6 unique indices.

    Args:
        parent1: First parent chromosome, shape (6,).
        parent2: Second parent chromosome, shape (6,).
        cache: Initialized TeambuildCache for replacement sampling.

    Returns:
        Tuple of (child1, child2), each shape (6,) with unique indices.
    """
    point = int(np.random.randint(1, 6))

    child1 = np.concatenate([parent1[:point], parent2[point:]])

    child2 = np.concatenate([parent2[:point], parent1[point:]])

    child1 = _deduplicate_child(child1, cache)
    child2 = _deduplicate_child(child2, cache)

    return child1, child2


def _deduplicate_child(
    child: NDArray[np.int64],
    cache: TeambuildCache,
) -> NDArray[np.int64]:
    """Replace duplicate indices in a child chromosome with unique values.

    Scans left to right; when a duplicate index is found, replaces it
    with a new sampled index not already present in the team.

    Args:
        child: Chromosome array of length 6, may contain duplicates.
        cache: Initialized TeambuildCache for replacement sampling.

    Returns:
        Deduplicated chromosome array of length 6 with all unique indices.
    """
    seen: set[int] = set()
    for i in range(len(child)):
        idx = int(child[i])
        if idx in seen:
            unused = np.setdiff1d(
                np.arange(cache.n_species, dtype=np.int64),
                np.array(list(seen), dtype=np.int64),
            )
            if len(unused) > 0:
                replacement = _weighted_sample_from_subset(cache, unused)
                child[i] = replacement
                seen.add(replacement)
        else:
            seen.add(idx)
    return child


def _weighted_sample_from_subset(
    cache: TeambuildCache,
    subset: NDArray[np.int64],
) -> int:
    """Sample one index from a subset, weighted by usage.

    Args:
        cache: Initialized TeambuildCache with usage_weights.
        subset: Array of candidate indices.

    Returns:
        A single integer index from subset.
    """
    if len(subset) == 0:
        return int(np.random.randint(0, cache.n_species))
    if len(subset) == 1:
        return int(subset[0])
    subset_weights = np.asarray(
        [cache.usage_weights[int(i)] for i in subset],
        dtype=np.float64,
    )
    total = subset_weights.sum()
    if total <= 0:
        return int(np.random.choice(subset))
    subset_weights /= total
    return int(np.random.choice(subset, p=subset_weights))


def mutate_team(
    team_indices: NDArray[np.int64],
    cache: TeambuildCache,
    mutation_rate: float,
) -> NDArray[np.int64]:
    """Apply per-position mutation to a team chromosome.

    For each of the 6 positions, with probability mutation_rate,
    replaces the species index with a new random species sampled
    from usage_weights. Enforces the no-duplicate constraint by
    only sampling from species not already in the team.

    Args:
        team_indices: Team chromosome, shape (6,).
        cache: Initialized TeambuildCache for replacement sampling.
        mutation_rate: Probability of mutating each position (0.0 to 1.0).

    Returns:
        Mutated team chromosome, shape (6,) with unique indices.
    """
    mutated = team_indices.copy()
    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            current_set = set(int(m) for m in mutated)
            unused = np.setdiff1d(
                np.arange(cache.n_species, dtype=np.int64),
                np.array(list(current_set - {int(mutated[i])}), dtype=np.int64),
            )
            if len(unused) > 0:
                mutated[i] = _weighted_sample_from_subset(cache, unused)
    return mutated
