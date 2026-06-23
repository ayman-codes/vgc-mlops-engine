"""Bayesian MAP fitness evaluator for team chromosome scoring.

Computes the surrogate fitness of a candidate team as the sum of
log-usage priors minus the minimum Euclidean distance to GMM archetype
centroids. Higher scores indicate teams that are both popular in the
metagame and aligned with a discovered archetype.
"""

import numpy as np

from src.agent.selection_policy.transformer import macro_features_array
from src.agent.teambuild_policy.cache import TeambuildCache

_EPSILON = 1e-10


def bayesian_map_fitness(
    indices: list[int],
    cache: TeambuildCache,
    archetype_weight: float = 1.0,
) -> float:
    """Compute Bayesian MAP fitness for a candidate team of 6 species.

    Fitness = Sum(log(P(Usage_i))) - Weight * Min_j(Distance(Features_scaled, Centroid_j))

    The usage term rewards teams composed of popular metagame species.
    The archetype distance term penalizes teams that do not resemble
    any discovered GMM archetype. The weight parameter scales the
    archetype distance to balance against usage log-priors.

    Args:
        indices: List of 6 integer indices into cache.species_keys.
        cache: Initialized TeambuildCache with GMM model and usage data.
        archetype_weight: Multiplier for the archetype distance penalty.
            Higher values penalize non-archetypal teams more heavily.

    Returns:
        Float fitness score. Higher is better. Negative values are
        possible when usage weights are very low or archetype distance
        is large.

    Raises:
        ValueError: If indices does not contain exactly 6 valid entries.
    """
    if len(indices) != 6:
        raise ValueError(
            f"bayesian_map_fitness requires exactly 6 indices, got {len(indices)}"
        )

    for idx in indices:
        if idx < 0 or idx >= cache.n_species:
            raise ValueError(
                f"Index {idx} out of range [0, {cache.n_species})"
            )

    usage_term = 0.0
    for i in indices:
        usage = max(cache.usage_weights[i], _EPSILON)
        usage_term += float(np.log(usage))

    species_names = [cache.species_keys[i] for i in indices]
    features = macro_features_array(species_names).reshape(1, -1)

    if cache.scaler is not None:
        features_scaled = cache.scaler.transform(features)
    else:
        features_scaled = features

    if cache.gmm is None:
        raise ValueError("GMM model is not loaded in the cache")

    centroids = cache.gmm.means_
    distances = np.linalg.norm(features_scaled - centroids, axis=1)
    archetype_distance = float(np.min(distances))

    return usage_term - (archetype_weight * archetype_distance)
