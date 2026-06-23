import numpy as np
import pytest

from src.agent.teambuild_policy.cache import TeambuildCache
from src.agent.teambuild_policy.operators import (
    _sample_unique_indices,
    _deduplicate_child,
    _weighted_sample_from_subset,
    init_population,
    crossover,
    mutate_team,
)


def _make_mock_cache(n_species: int = 20) -> TeambuildCache:
    """Build a minimal TeambuildCache with uniform usage weights.

    Args:
        n_species: Number of species in the cache pool.

    Returns:
        TeambuildCache with uniform usage_weights and empty raw_data/GMM.
    """
    usage_vals = np.ones(n_species, dtype=np.float64)
    usage_weights = usage_vals / usage_vals.sum()

    return TeambuildCache(
        species_keys=[f"species_{i}" for i in range(n_species)],
        usage_weights=usage_weights,
        raw_data={},
        gmm=None,
        scaler=None,
    )


def _all_unique(arr: np.ndarray) -> bool:
    """Check if all values in a 1D array are unique."""
    return len(set(arr.tolist())) == len(arr)


class TestSampleUniqueIndices:
    """Tests for _sample_unique_indices."""

    def test_returns_correct_length(self) -> None:
        cache = _make_mock_cache(20)
        result = _sample_unique_indices(cache, n=6)
        assert len(result) == 6

    def test_all_unique(self) -> None:
        cache = _make_mock_cache(20)
        for _ in range(100):
            result = _sample_unique_indices(cache, n=6)
            assert _all_unique(result), f"Duplicate found: {result}"

    def test_indices_in_valid_range(self) -> None:
        cache = _make_mock_cache(20)
        for _ in range(100):
            result = _sample_unique_indices(cache, n=6)
            assert np.all(result >= 0)
            assert np.all(result < 20)

    def test_raises_when_pool_too_small(self) -> None:
        cache = _make_mock_cache(4)
        with pytest.raises(ValueError):
            _sample_unique_indices(cache, n=6)

    def test_different_calls_produce_different_samples(self) -> None:
        cache = _make_mock_cache(50)
        results = [_sample_unique_indices(cache, n=6) for _ in range(20)]
        arrays = [tuple(sorted(r)) for r in results]
        unique_count = len(set(arrays))
        assert unique_count >= 10, (
            f"Expected diverse samples, got {unique_count}/20 unique"
        )


class TestDeduplicateChild:
    """Tests for _deduplicate_child."""

    def test_no_op_when_already_unique(self) -> None:
        cache = _make_mock_cache(20)
        child = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
        result = _deduplicate_child(child, cache)
        assert np.array_equal(result, child)

    def test_replaces_duplicates(self) -> None:
        cache = _make_mock_cache(20)
        child = np.array([0, 1, 2, 0, 4, 5], dtype=np.int64)
        result = _deduplicate_child(child, cache)
        assert _all_unique(result)

    def test_all_duplicates_replaced(self) -> None:
        cache = _make_mock_cache(20)
        child = np.array([0, 0, 0, 0, 0, 0], dtype=np.int64)
        result = _deduplicate_child(child, cache)
        assert _all_unique(result)

    def test_keeps_length(self) -> None:
        cache = _make_mock_cache(20)
        child = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        result = _deduplicate_child(child, cache)
        assert len(result) == 6


class TestWeightedSampleFromSubset:
    """Tests for _weighted_sample_from_subset."""

    def test_returns_from_subset(self) -> None:
        cache = _make_mock_cache(20)
        subset = np.array([5, 6, 7], dtype=np.int64)
        for _ in range(20):
            result = _weighted_sample_from_subset(cache, subset)
            assert result in subset

    def test_empty_subset_returns_valid_index(self) -> None:
        cache = _make_mock_cache(20)
        result = _weighted_sample_from_subset(cache, np.array([], dtype=np.int64))
        assert 0 <= result < 20

    def test_single_element_subset(self) -> None:
        cache = _make_mock_cache(20)
        result = _weighted_sample_from_subset(cache, np.array([7], dtype=np.int64))
        assert result == 7


class TestInitPopulation:
    """Tests for init_population."""

    def test_returns_correct_shape(self) -> None:
        cache = _make_mock_cache(20)
        pop = init_population(10, cache)
        assert pop.shape == (10, 6)
        assert pop.dtype == np.int64

    def test_all_teams_have_unique_indices(self) -> None:
        cache = _make_mock_cache(50)
        pop = init_population(100, cache)
        for i in range(100):
            assert _all_unique(pop[i]), (
                f"Team {i} has duplicates: {pop[i]}"
            )

    def test_all_indices_in_valid_range(self) -> None:
        cache = _make_mock_cache(30)
        pop = init_population(100, cache)
        assert np.all(pop >= 0)
        assert np.all(pop < 30)

    def test_different_seeds_produce_different_populations(self) -> None:
        np.random.seed(42)
        cache = _make_mock_cache(100)
        pop1 = init_population(10, cache)
        np.random.seed(99)
        pop2 = init_population(10, cache)
        assert not np.array_equal(pop1, pop2)

    def test_seeded_produces_deterministic(self) -> None:
        cache = _make_mock_cache(100)
        np.random.seed(42)
        pop1 = init_population(10, cache)
        np.random.seed(42)
        pop2 = init_population(10, cache)
        assert np.array_equal(pop1, pop2)

    def test_zero_population_returns_empty(self) -> None:
        cache = _make_mock_cache(20)
        pop = init_population(0, cache)
        assert pop.shape == (0, 6)

    def test_raises_when_too_few_species(self) -> None:
        cache = _make_mock_cache(3)
        with pytest.raises(ValueError):
            init_population(10, cache)

    def test_weighted_sampling_favors_high_weight_species(self) -> None:
        """Species with higher usage weights appear more frequently."""
        n_species = 10
        usage_vals = np.array([0.01] + [1.0] * (n_species - 1), dtype=np.float64)
        usage_weights = usage_vals / usage_vals.sum()

        cache = TeambuildCache(
            species_keys=[f"species_{i}" for i in range(n_species)],
            usage_weights=usage_weights,
            raw_data={},
            gmm=None,
            scaler=None,
        )

        pop = init_population(500, cache)
        all_indices = pop.flatten()
        count_0 = int(np.sum(all_indices == 0))
        assert count_0 < 100, (
            f"Low-weight species 0 should appear rarely, got {count_0}/3000"
        )


class TestCrossover:
    """Tests for crossover."""

    def test_returns_two_children(self) -> None:
        cache = _make_mock_cache(50)
        p1 = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
        p2 = np.array([5, 6, 7, 8, 9, 10], dtype=np.int64)
        child1, child2 = crossover(p1, p2, cache)
        assert child1.shape == (6,)
        assert child2.shape == (6,)

    def test_children_have_unique_indices(self) -> None:
        cache = _make_mock_cache(50)
        p1 = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
        p2 = np.array([5, 4, 3, 2, 1, 0], dtype=np.int64)
        for _ in range(100):
            child1, child2 = crossover(p1, p2, cache)
            assert _all_unique(child1), f"Child1 has duplicates: {child1}"
            assert _all_unique(child2), f"Child2 has duplicates: {child2}"

    def test_children_contain_parent_elements(self) -> None:
        cache = _make_mock_cache(50)
        p1 = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
        p2 = np.array([6, 7, 8, 9, 10, 11], dtype=np.int64)
        parent_union = set(p1.tolist()) | set(p2.tolist())
        for _ in range(20):
            child1, child2 = crossover(p1, p2, cache)
            c1_set = set(child1.tolist())
            c2_set = set(child2.tolist())
            assert len(c1_set & parent_union) >= 2
            assert len(c2_set & parent_union) >= 2

    def test_deterministic_with_seed(self) -> None:
        cache = _make_mock_cache(50)
        p1 = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
        p2 = np.array([5, 4, 3, 2, 1, 0], dtype=np.int64)
        np.random.seed(42)
        c1a, c2a = crossover(p1, p2, cache)
        np.random.seed(42)
        c1b, c2b = crossover(p1, p2, cache)
        assert np.array_equal(c1a, c1b)
        assert np.array_equal(c2a, c2b)


class TestMutateTeam:
    """Tests for mutate_team."""

    def test_zero_rate_returns_unchanged(self) -> None:
        cache = _make_mock_cache(20)
        team = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
        for _ in range(20):
            result = mutate_team(team, cache, 0.0)
            assert np.array_equal(result, team)

    def test_unique_after_mutation(self) -> None:
        cache = _make_mock_cache(50)
        team = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
        for _ in range(100):
            result = mutate_team(team, cache, 0.3)
            assert _all_unique(result)

    def test_high_rate_produces_some_change(self) -> None:
        cache = _make_mock_cache(50)
        team = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
        changes = 0
        for _ in range(50):
            result = mutate_team(team, cache, 0.9)
            if not np.array_equal(result, team):
                changes += 1
        assert changes >= 30, (
            f"With rate 0.9, expected most runs to mutate; got {changes}/50"
        )

    def test_full_rate_changes_all(self) -> None:
        cache = _make_mock_cache(100)
        team = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
        results = []
        for _ in range(20):
            result = mutate_team(team, cache, 1.0)
            assert _all_unique(result)
            results.append(result)
        identities = sum(1 for r in results if np.array_equal(r, team))
        assert identities == 0, (
            "With rate 1.0, team should never be identical"
        )

    def test_indices_in_valid_range(self) -> None:
        cache = _make_mock_cache(30)
        team = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
        for _ in range(50):
            result = mutate_team(team, cache, 0.5)
            assert np.all(result >= 0)
            assert np.all(result < 30)

    def test_deterministic_with_seed(self) -> None:
        cache = _make_mock_cache(50)
        team = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
        np.random.seed(42)
        r1 = mutate_team(team, cache, 0.5)
        np.random.seed(42)
        r2 = mutate_team(team, cache, 0.5)
        assert np.array_equal(r1, r2)
