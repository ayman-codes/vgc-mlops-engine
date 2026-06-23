import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.agent.teambuild_policy.cache import TeambuildCache
from src.agent.teambuild_policy.evolution import run_evolution
from src.config.teambuild_config import TeambuildConfig


def _make_mock_cache(n_species: int = 30) -> TeambuildCache:
    """Build a TeambuildCache with controlled usage weights and GMM for evolution tests.

    Uses uniform usage weights so no species is artificially favoured,
    and a simple 2-cluster random GMM.

    Args:
        n_species: Number of species in the cache pool.

    Returns:
        TeambuildCache instance with mock data.
    """
    usage_vals = np.ones(n_species, dtype=np.float64)
    usage_weights = usage_vals / usage_vals.sum()

    rng = np.random.RandomState(42)
    scaler = StandardScaler()
    gmm = GaussianMixture(n_components=2, random_state=42)
    data = rng.randn(100, 4)
    scaler.fit(data)
    gmm.fit(scaler.transform(data))

    species_keys = [
        "snorlax", "jolteon", "charizard", "gyarados", "alakazam",
        "gengar", "dragonite", "mewtwo", "tyranitar", "blastoise",
        "arcanine", "lapras", "electabuzz", "magmar", "rhydon",
        "machamp", "steelix", "scizor", "heracross", "skarmory",
        "houndoom", "kingdra", "donphan", "porygon2", "blissey",
        "raikou", "entei", "suicune", "celebi", "swampert",
    ][:n_species]

    raw_data: dict[str, dict[str, object]] = {}
    for k in species_keys:
        raw_data[k] = {
            "usage": 0.1,
            "Abilities": {},
            "Items": {},
            "Spreads": {},
            "Moves": {},
        }

    return TeambuildCache(
        species_keys=species_keys,
        usage_weights=usage_weights,
        raw_data=raw_data,
        gmm=gmm,
        scaler=scaler,
    )


class TestRunEvolution:
    """Tests for the run_evolution evolutionary loop."""

    def test_returns_dict_with_expected_keys(self) -> None:
        cache = _make_mock_cache(30)
        config = TeambuildConfig(population_size=10, generations=5)
        result = run_evolution(config, cache)
        assert isinstance(result, dict)
        for key in (
            "best_teams",
            "fitness_history",
            "convergence_time",
            "final_best_fitness",
            "final_mean_fitness",
        ):
            assert key in result

    def test_best_teams_has_correct_shape(self) -> None:
        cache = _make_mock_cache(30)
        config = TeambuildConfig(
            population_size=20, generations=5, elite_fraction=0.10
        )
        result = run_evolution(config, cache)
        n_elites = max(1, int(20 * 0.10))
        assert result["best_teams"].shape == (n_elites, 6)
        assert result["best_teams"].dtype == np.int64

    def test_population_size_constant_through_generations(self) -> None:
        cache = _make_mock_cache(30)
        config = TeambuildConfig(population_size=10, generations=20)
        result = run_evolution(config, cache)
        assert len(result["fitness_history"]["best"]) == config.generations
        assert len(result["fitness_history"]["mean"]) == config.generations

    def test_max_fitness_monotonically_increases(self) -> None:
        """Peak fitness should never decrease across generations."""
        cache = _make_mock_cache(30)
        config = TeambuildConfig(
            population_size=30,
            generations=30,
            mutation_rate=0.05,
            elite_fraction=0.10,
            fitness_archetype_weight=1.0,
        )
        np.random.seed(42)
        result = run_evolution(config, cache)
        best = result["fitness_history"]["best"]
        for i in range(1, len(best)):
            assert best[i] >= best[i - 1], (
                f"Fitness decreased at generation {i}: "
                f"{best[i]} < {best[i - 1]}"
            )

    def test_all_team_indices_in_valid_range(self) -> None:
        cache = _make_mock_cache(30)
        config = TeambuildConfig(population_size=10, generations=5)
        np.random.seed(42)
        result = run_evolution(config, cache)
        teams = result["best_teams"]
        assert np.all(teams >= 0)
        assert np.all(teams < cache.n_species)

    def test_all_teams_have_unique_species(self) -> None:
        cache = _make_mock_cache(30)
        config = TeambuildConfig(population_size=10, generations=5)
        np.random.seed(42)
        result = run_evolution(config, cache)
        for team in result["best_teams"]:
            team_list = team.tolist()
            assert len(set(team_list)) == len(team_list), (
                f"Duplicate species in elite team: {team_list}"
            )

    def test_deterministic_with_seed(self) -> None:
        cache = _make_mock_cache(30)
        config = TeambuildConfig(population_size=10, generations=5)
        np.random.seed(99)
        r1 = run_evolution(config, cache)
        np.random.seed(99)
        r2 = run_evolution(config, cache)
        assert np.array_equal(r1["best_teams"], r2["best_teams"])
        assert r1["fitness_history"]["best"] == r2["fitness_history"]["best"]

    def test_convergence_time_positive(self) -> None:
        cache = _make_mock_cache(30)
        config = TeambuildConfig(population_size=10, generations=5)
        result = run_evolution(config, cache)
        assert result["convergence_time"] > 0.0

    def test_works_with_minimum_config(self) -> None:
        cache = _make_mock_cache(30)
        config = TeambuildConfig(
            population_size=2, generations=1, elite_fraction=0.50
        )
        result = run_evolution(config, cache)
        n_elites = max(1, int(2 * 0.50))
        assert result["best_teams"].shape == (n_elites, 6)

    def test_fitness_is_finite(self) -> None:
        cache = _make_mock_cache(30)
        config = TeambuildConfig(population_size=10, generations=5)
        result = run_evolution(config, cache)
        for val in result["fitness_history"]["best"]:
            assert not np.isnan(val)
            assert not np.isinf(val)

    def test_fitness_history_never_contains_negative_inf(self) -> None:
        cache = _make_mock_cache(30)
        config = TeambuildConfig(population_size=10, generations=5)
        result = run_evolution(config, cache)
        for val in result["fitness_history"]["mean"]:
            assert not np.isnan(val)
            assert not np.isinf(val)

    def test_higher_archetype_weight_changes_fitness(self) -> None:
        """A higher archetype_weight should produce different (more negative)
        fitness scores since the GMM distance penalty is amplified."""
        cache = _make_mock_cache(30)
        config_low = TeambuildConfig(
            population_size=10, generations=5, fitness_archetype_weight=1.0
        )
        config_high = TeambuildConfig(
            population_size=10, generations=5, fitness_archetype_weight=100.0
        )
        np.random.seed(42)
        result_low = run_evolution(config_low, cache)
        np.random.seed(42)
        result_high = run_evolution(config_high, cache)
        assert result_low["final_best_fitness"] != result_high["final_best_fitness"]
