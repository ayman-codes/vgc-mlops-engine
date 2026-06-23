from unittest.mock import patch, MagicMock

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.agent.teambuild_policy.cache import TeambuildCache
from src.agent.teambuilder import EvolutionaryTeambuilder
from src.config.teambuild_config import TeambuildConfig


class FakeMLflowRun:
    """Mock context manager that does NOT suppress exceptions."""

    def __enter__(self) -> "FakeMLflowRun":
        return self

    def __exit__(self, *args: object) -> bool:
        return False


def _make_mock_cache(n_species: int = 30) -> TeambuildCache:
    """Build a TeambuildCache for pipeline integration testing.

    Small realistic dataset with real Pokemon names for Showdown
    serialization.

    Args:
        n_species: Number of species in the cache pool.

    Returns:
        TeambuildCache instance.
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
            "Abilities": {"torrent": 1.0},
            "Items": {"leftovers": 1.0},
            "Spreads": {
                "Calm:252/0/4/0/252/0": 1.0,
            },
            "Moves": {
                "tackle": 1.0,
                "growl": 0.8,
                "watergun": 0.6,
                "protect": 0.4,
            },
        }

    return TeambuildCache(
        species_keys=species_keys,
        usage_weights=usage_weights,
        raw_data=raw_data,
        gmm=gmm,
        scaler=scaler,
    )


def _make_mock_royale_result(best_team_indices: list[int]) -> dict[str, object]:
    """Create a mock return value for run_battle_royale.

    Returns a dict matching the real function's output shape.
    """
    return {
        "best_team_idx": 0,
        "best_team_indices": best_team_indices,
        "empirical_win_rate": 0.75,
        "all_win_rates": {0: 0.75},
    }


class TestEvolutionaryTeambuilderPipeline:
    """End-to-end integration tests for the EvolutionaryTeambuilder."""

    def test_yield_team_returns_non_empty_string(self) -> None:
        """Full pipeline should return a non-empty Showdown team string."""
        cache = _make_mock_cache(30)
        config = TeambuildConfig(
            population_size=10,
            generations=5,
            elite_fraction=0.10,
            fitness_archetype_weight=1.0,
            battle_royale_n=1,
        )
        builder = EvolutionaryTeambuilder(config=config, cache=cache)

        mock_royale_result = _make_mock_royale_result([0, 1, 2, 3, 4, 5])

        with patch(
            "src.agent.teambuilder.asyncio.run",
            side_effect=lambda x: x,
        ), patch(
            "src.agent.teambuilder.run_battle_royale",
            return_value=mock_royale_result,
        ), patch(
            "src.agent.teambuilder.mlflow.start_run",
            return_value=FakeMLflowRun(),
        ), patch(
            "src.agent.teambuilder.mlflow.log_params",
            MagicMock(),
        ), patch(
            "src.agent.teambuilder.mlflow.log_metric",
            MagicMock(),
        ):
            team_str = builder.yield_team()

        assert isinstance(team_str, str)
        assert len(team_str) > 0
        assert "\n" in team_str

    def test_yield_team_contains_expected_format(self) -> None:
        """Returned team string should contain Pokemon Showdown format tokens."""
        cache = _make_mock_cache(30)
        config = TeambuildConfig(
            population_size=10,
            generations=5,
            fitness_archetype_weight=1.0,
            battle_royale_n=1,
        )
        builder = EvolutionaryTeambuilder(config=config, cache=cache)

        mock_royale_result = _make_mock_royale_result([0, 1, 2, 3, 4, 5])

        with patch(
            "src.agent.teambuilder.asyncio.run",
            side_effect=lambda x: x,
        ), patch(
            "src.agent.teambuilder.run_battle_royale",
            return_value=mock_royale_result,
        ), patch(
            "src.agent.teambuilder.mlflow.start_run",
            return_value=FakeMLflowRun(),
        ), patch(
            "src.agent.teambuilder.mlflow.log_params",
            MagicMock(),
        ), patch(
            "src.agent.teambuilder.mlflow.log_metric",
            MagicMock(),
        ):
            team_str = builder.yield_team()

        assert "EVs:" in team_str or "Level:" in team_str
        assert "- " in team_str

    def test_yield_team_returns_six_pokemon_blocks(self) -> None:
        """Team string should contain 6 Pokemon blocks separated by
        double newlines."""
        cache = _make_mock_cache(30)
        config = TeambuildConfig(
            population_size=10,
            generations=5,
            fitness_archetype_weight=1.0,
            battle_royale_n=1,
        )
        builder = EvolutionaryTeambuilder(config=config, cache=cache)

        mock_royale_result = _make_mock_royale_result([0, 1, 2, 3, 4, 5])

        with patch(
            "src.agent.teambuilder.asyncio.run",
            side_effect=lambda x: x,
        ), patch(
            "src.agent.teambuilder.run_battle_royale",
            return_value=mock_royale_result,
        ), patch(
            "src.agent.teambuilder.mlflow.start_run",
            return_value=FakeMLflowRun(),
        ), patch(
            "src.agent.teambuilder.mlflow.log_params",
            MagicMock(),
        ), patch(
            "src.agent.teambuilder.mlflow.log_metric",
            MagicMock(),
        ):
            team_str = builder.yield_team()

        blocks = [b for b in team_str.split("\n\n") if b.strip()]
        assert len(blocks) >= 6

    def test_yield_team_logs_mlflow_metrics(self) -> None:
        """Verify mlflow.log_metric is called with expected metric names."""
        cache = _make_mock_cache(30)
        config = TeambuildConfig(
            population_size=10,
            generations=3,
            battle_royale_n=1,
        )
        builder = EvolutionaryTeambuilder(config=config, cache=cache)

        mock_log_metric = MagicMock()
        mock_royale_result = _make_mock_royale_result([0, 1, 2, 3, 4, 5])

        with patch(
            "src.agent.teambuilder.asyncio.run",
            side_effect=lambda x: x,
        ), patch(
            "src.agent.teambuilder.run_battle_royale",
            return_value=mock_royale_result,
        ), patch(
            "src.agent.teambuilder.mlflow.start_run",
            return_value=FakeMLflowRun(),
        ), patch(
            "src.agent.teambuilder.mlflow.log_params",
            MagicMock(),
        ), patch(
            "src.agent.teambuilder.mlflow.log_metric",
            mock_log_metric,
        ):
            builder.yield_team()

        logged_metrics = {
            call_args[0][0]
            for call_args in mock_log_metric.call_args_list
            if len(call_args[0]) >= 1
        }
        assert "map_fitness_peak" in logged_metrics
        assert "map_fitness_mean" in logged_metrics
        assert "generation_convergence_time" in logged_metrics
        assert "empirical_win_rate" in logged_metrics

    def test_yield_team_handles_cache_with_fewer_species(self) -> None:
        """With exactly 6 species, the pipeline should still work."""
        cache = _make_mock_cache(6)
        config = TeambuildConfig(
            population_size=10,
            generations=3,
            battle_royale_n=1,
        )
        builder = EvolutionaryTeambuilder(config=config, cache=cache)

        mock_royale_result = _make_mock_royale_result([0, 1, 2, 3, 4, 5])

        with patch(
            "src.agent.teambuilder.asyncio.run",
            side_effect=lambda x: x,
        ), patch(
            "src.agent.teambuilder.run_battle_royale",
            return_value=mock_royale_result,
        ), patch(
            "src.agent.teambuilder.mlflow.start_run",
            return_value=FakeMLflowRun(),
        ), patch(
            "src.agent.teambuilder.mlflow.log_params",
            MagicMock(),
        ), patch(
            "src.agent.teambuilder.mlflow.log_metric",
            MagicMock(),
        ):
            team_str = builder.yield_team()

        assert len(team_str) > 0
        assert "\n" in team_str
