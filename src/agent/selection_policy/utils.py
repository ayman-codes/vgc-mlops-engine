"""Shared utilities for the V2 selection policy pipeline.

Pure functions for team order serialization and Shannon entropy
computation, imported by the entrypoint, agent, and benchmarks.
"""

import numpy as np
from numpy.typing import NDArray

from src.agent.selection_policy.double_oracle import (
    Strategy,
    _species_showdown_name,
)


def build_team_order(species: list[str], strategy: Strategy) -> str:
    """Build a Showdown /team command from species list and strategy indices.

    Args:
        species: List of 6 species names (lowercase).
        strategy: 6-element index tuple (lead1, lead2, back1, back2,
            back3, back4).

    Returns:
        Showdown /team command string, e.g. "/team snorlax|charizard|...".
    """
    showdown_names = [_species_showdown_name(species[i]) for i in strategy]
    return "/team " + "|".join(showdown_names)


def shannon_entropy(probs: NDArray[np.float64]) -> float:
    """Compute Shannon entropy from a probability vector.

    Args:
        probs: Probability distribution vector.

    Returns:
        Entropy value in bits. Zero for deterministic distributions.
    """
    p = probs[probs > 0]
    return float(-np.sum(p * np.log2(p)))
