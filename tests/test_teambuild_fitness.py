import numpy as np
import pytest
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.agent.teambuild_policy.cache import TeambuildCache
from src.agent.teambuild_policy.fitness import bayesian_map_fitness


def _make_mock_cache() -> TeambuildCache:
    """Build a minimal TeambuildCache with controlled usage weights and GMM.

    Creates 10 species with evenly spaced usage weights and a 2-cluster
    GMM with known centroids. The mock raw_data uses simplified entries
    that poke_env.GenData can resolve.

    Returns:
        TeambuildCache instance ready for fitness testing.
    """
    n_species = 10
    usage_vals = np.linspace(0.01, 0.20, n_species, dtype=np.float64)
    usage_weights = usage_vals / usage_vals.sum()

    rng = np.random.RandomState(42)
    scaler = StandardScaler()
    gmm = GaussianMixture(n_components=2, random_state=42)
    data = rng.randn(50, 4)
    scaler.fit(data)
    gmm.fit(scaler.transform(data))

    species_keys = [
        "snorlax", "jolteon", "charizard", "gyarados",
        "alakazam", "gengar", "dragonite", "mewtwo",
        "tyranitar", "blastoise",
    ][:n_species]

    raw_data: dict[str, dict] = {}
    for k in species_keys:
        raw_data[k] = {"usage": 0.1, "Abilities": {}, "Items": {}, "Spreads": {}, "Moves": {}}

    return TeambuildCache(
        species_keys=species_keys,
        usage_weights=usage_weights,
        raw_data=raw_data,
        gmm=gmm,
        scaler=scaler,
    )


class TestBayesianMapFitness:
    """Tests for the bayesian_map_fitness function."""

    def test_returns_float(self) -> None:
        cache = _make_mock_cache()
        result = bayesian_map_fitness([0, 1, 2, 3, 4, 5], cache)
        assert isinstance(result, float)

    def test_deterministic_output(self) -> None:
        cache = _make_mock_cache()
        r1 = bayesian_map_fitness([0, 1, 2, 3, 4, 5], cache)
        r2 = bayesian_map_fitness([0, 1, 2, 3, 4, 5], cache)
        assert r1 == r2

    def test_raises_on_wrong_length(self) -> None:
        cache = _make_mock_cache()
        with pytest.raises(ValueError, match="exactly 6"):
            bayesian_map_fitness([0, 1, 2], cache)
        with pytest.raises(ValueError, match="exactly 6"):
            bayesian_map_fitness([0, 1, 2, 3, 4, 5, 6], cache)

    def test_raises_on_negative_index(self) -> None:
        cache = _make_mock_cache()
        with pytest.raises(ValueError, match="out of range"):
            bayesian_map_fitness([-1, 0, 1, 2, 3, 4], cache)

    def test_raises_on_out_of_range_index(self) -> None:
        cache = _make_mock_cache()
        with pytest.raises(ValueError, match="out of range"):
            bayesian_map_fitness([0, 1, 2, 3, 4, 99], cache)

    def test_identical_indices_same_score(self) -> None:
        """Identical indices produce identical fitness scores."""
        cache = _make_mock_cache()
        score_a = bayesian_map_fitness([0, 1, 2, 3, 4, 5], cache)
        score_b = bayesian_map_fitness([0, 1, 2, 3, 4, 5], cache)
        assert score_a == score_b

    def test_same_species_composition_same_gmm_component(self) -> None:
        """Teams with identical species composition differ only by usage
        log-priors. The GMM distance component is identical because the
        macro-features are the same."""
        cache = _make_mock_cache()
        score_ordered = bayesian_map_fitness([0, 1, 2, 3, 4, 5], cache)
        score_shuffled = bayesian_map_fitness([5, 4, 3, 2, 1, 0], cache)
        assert score_ordered == score_shuffled, (
            "Macro features are aggregate, order-independent; "
            "usage_sum(log(weights)) is also order-independent"
        )

    def test_raises_when_gmm_is_none(self) -> None:
        cache = _make_mock_cache()
        cache.gmm = None
        with pytest.raises(ValueError, match="GMM model is not loaded"):
            bayesian_map_fitness([0, 1, 2, 3, 4, 5], cache)

    def test_works_without_scaler(self) -> None:
        cache = _make_mock_cache()
        cache.scaler = None
        result = bayesian_map_fitness([0, 1, 2, 3, 4, 5], cache)
        assert isinstance(result, float)

    def test_empty_indices_raises(self) -> None:
        cache = _make_mock_cache()
        with pytest.raises(ValueError, match="exactly 6"):
            bayesian_map_fitness([], cache)

    def test_fitness_is_not_inf_or_nan(self) -> None:
        cache = _make_mock_cache()
        indices = [0, 1, 2, 3, 4, 5]
        result = bayesian_map_fitness(indices, cache)
        assert not np.isnan(result)
        assert not np.isinf(result)

    def test_fitness_allows_negative_values(self) -> None:
        """With low enough usage, fitness can be negative due to log penalty."""
        n_species = 10
        usage_vals = np.array(
            [1e-10] * n_species, dtype=np.float64
        )
        usage_weights = usage_vals / usage_vals.sum()

        rng = np.random.RandomState(42)
        scaler = StandardScaler()
        gmm = GaussianMixture(n_components=2, random_state=42)
        data = rng.randn(50, 4)
        scaler.fit(data)
        gmm.fit(scaler.transform(data))

        species = ["snorlax"] * n_species
        cache = TeambuildCache(
            species_keys=species,
            usage_weights=usage_weights,
            raw_data={},
            gmm=gmm,
            scaler=scaler,
        )

        result = bayesian_map_fitness([0, 1, 2, 3, 4, 5], cache)
        assert result < 0.0, f"Expected negative fitness, got {result}"
