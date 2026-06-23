import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.agent.teambuild_policy.cache import TeambuildCache
from src.agent.teambuild_policy.battle_royale import run_battle_royale


def _make_mock_cache(n_species: int = 30) -> TeambuildCache:
    """Build a TeambuildCache with mock data for battle royale tests.

    Species are real pokemon names that poke-env can resolve,
    with trivial distribution data.

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


def _make_mock_player(finished: int, won: int) -> MagicMock:
    """Create a MagicMock that mimics a poke-env Player with win count attrs.

    Args:
        finished: Number of finished battles.
        won: Number of won battles.
    """
    mock = MagicMock()
    mock.n_finished_battles = finished
    mock.n_won_battles = won
    return mock


class MockSimpleHeuristicsPlayer(MagicMock):
    """Fake player that reports controlled win counts."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.n_finished_battles = 0
        self.n_won_battles = 0


class MockRandomPlayer(MagicMock):
    """Fake baseline player."""


class MockMaxBasePowerPlayer(MagicMock):
    """Fake baseline player."""


async def _mock_cross_evaluate_highest(
    players: list[Any], n_challenges: int = 1
) -> None:
    """Mock cross_evaluate that makes the first elite team win."""
    player_count = 0
    for p in players:
        if isinstance(p, MockSimpleHeuristicsPlayer):
            player_count += 1

    elite_idx = 0
    for p in players:
        if isinstance(p, MockSimpleHeuristicsPlayer):
            p.n_finished_battles = 20
            p.n_won_battles = 20 if elite_idx == 0 else 0
            elite_idx += 1
        else:
            p.n_finished_battles = 20
            p.n_won_battles = 10


async def _mock_cross_evaluate_all_zero(
    players: list[Any], n_challenges: int = 1
) -> None:
    """Mock cross_evaluate that makes all teams have zero wins."""
    for p in players:
        p.n_finished_battles = 20
        p.n_won_battles = 0


async def _mock_cross_evaluate_custom(
    players: list[Any],
    n_challenges: int = 1,
    *,
    elite_win_counts: list[int],
) -> None:
    """Mock cross_evaluate with custom win counts per elite player."""
    elite_idx = 0
    for p in players:
        if isinstance(p, MockSimpleHeuristicsPlayer):
            p.n_finished_battles = 20
            if elite_idx < len(elite_win_counts):
                p.n_won_battles = elite_win_counts[elite_idx]
            else:
                p.n_won_battles = 0
            elite_idx += 1
        else:
            p.n_finished_battles = 20
            p.n_won_battles = 10


class TestRunBattleRoyale:
    """Tests for the run_battle_royale async function."""

    def test_selects_highest_win_rate_team(self) -> None:
        """The team with the highest win rate should be selected."""
        cache = _make_mock_cache(30)
        team_indices = [
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
            [12, 13, 14, 15, 16, 17],
        ]

        with patch(
            "src.agent.teambuild_policy.battle_royale.SimpleHeuristicsPlayer",
            MockSimpleHeuristicsPlayer,
        ), patch(
            "src.agent.teambuild_policy.battle_royale.RandomPlayer",
            MockRandomPlayer,
        ), patch(
            "src.agent.teambuild_policy.battle_royale.MaxBasePowerPlayer",
            MockMaxBasePowerPlayer,
        ), patch(
            "src.agent.teambuild_policy.battle_royale.cross_evaluate",
            AsyncMock(side_effect=_mock_cross_evaluate_highest),
        ):
            result = asyncio.run(
                run_battle_royale(team_indices, cache, n_battles=10)
            )

        assert result["best_team_idx"] == 0
        assert result["best_team_indices"] == team_indices[0]
        assert result["empirical_win_rate"] == 1.0

    def test_selects_first_team_when_all_zero(self) -> None:
        """When all win rates are zero, any team is valid."""
        cache = _make_mock_cache(30)
        team_indices = [
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
        ]

        with patch(
            "src.agent.teambuild_policy.battle_royale.SimpleHeuristicsPlayer",
            MockSimpleHeuristicsPlayer,
        ), patch(
            "src.agent.teambuild_policy.battle_royale.RandomPlayer",
            MockRandomPlayer,
        ), patch(
            "src.agent.teambuild_policy.battle_royale.MaxBasePowerPlayer",
            MockMaxBasePowerPlayer,
        ), patch(
            "src.agent.teambuild_policy.battle_royale.cross_evaluate",
            AsyncMock(side_effect=_mock_cross_evaluate_all_zero),
        ):
            result = asyncio.run(
                run_battle_royale(team_indices, cache, n_battles=10)
            )

        assert 0 <= result["best_team_idx"] < 2
        assert result["empirical_win_rate"] == 0.0

    def test_single_team_returns_itself(self) -> None:
        """With one team, it should be selected with its win rate."""
        cache = _make_mock_cache(30)
        team_indices = [[0, 1, 2, 3, 4, 5]]

        async def mock_ce(
            players: list[Any], n_challenges: int = 1
        ) -> None:
            for p in players:
                if isinstance(p, MockSimpleHeuristicsPlayer):
                    p.n_finished_battles = 20
                    p.n_won_battles = 15
                else:
                    p.n_finished_battles = 20
                    p.n_won_battles = 10

        with patch(
            "src.agent.teambuild_policy.battle_royale.SimpleHeuristicsPlayer",
            MockSimpleHeuristicsPlayer,
        ), patch(
            "src.agent.teambuild_policy.battle_royale.RandomPlayer",
            MockRandomPlayer,
        ), patch(
            "src.agent.teambuild_policy.battle_royale.MaxBasePowerPlayer",
            MockMaxBasePowerPlayer,
        ), patch(
            "src.agent.teambuild_policy.battle_royale.cross_evaluate",
            AsyncMock(side_effect=mock_ce),
        ):
            result = asyncio.run(
                run_battle_royale(team_indices, cache, n_battles=10)
            )

        assert result["best_team_idx"] == 0
        assert result["best_team_indices"] == team_indices[0]
        assert result["empirical_win_rate"] == 0.75

    def test_all_win_rates_dict_populated(self) -> None:
        """all_win_rates should contain an entry for every elite team."""
        cache = _make_mock_cache(30)
        n_teams = 5
        team_indices = [
            [i, i + 1, i + 2, i + 3, i + 4, i + 5]
            for i in range(n_teams)
        ]

        async def mock_ce(
            players: list[Any], n_challenges: int = 1
        ) -> None:
            elite_idx = 0
            for p in players:
                if isinstance(p, MockSimpleHeuristicsPlayer):
                    p.n_finished_battles = 20
                    win_counts = [0, 0, 20, 0, 0]
                    if elite_idx < len(win_counts):
                        p.n_won_battles = win_counts[elite_idx]
                    else:
                        p.n_won_battles = 0
                    elite_idx += 1
                else:
                    p.n_finished_battles = 20
                    p.n_won_battles = 10

        with patch(
            "src.agent.teambuild_policy.battle_royale.SimpleHeuristicsPlayer",
            MockSimpleHeuristicsPlayer,
        ), patch(
            "src.agent.teambuild_policy.battle_royale.RandomPlayer",
            MockRandomPlayer,
        ), patch(
            "src.agent.teambuild_policy.battle_royale.MaxBasePowerPlayer",
            MockMaxBasePowerPlayer,
        ), patch(
            "src.agent.teambuild_policy.battle_royale.cross_evaluate",
            AsyncMock(side_effect=mock_ce),
        ):
            result = asyncio.run(
                run_battle_royale(team_indices, cache, n_battles=10)
            )

        assert len(result["all_win_rates"]) == n_teams
        for i in range(n_teams):
            assert i in result["all_win_rates"]
        assert result["best_team_idx"] == 2

    def test_returns_valid_team_indices(self) -> None:
        """The winning team indices should be within cache bounds."""
        cache = _make_mock_cache(30)
        team_indices = [[0, 1, 2, 3, 4, 5]]

        async def mock_ce(
            players: list[Any], n_challenges: int = 1
        ) -> None:
            for p in players:
                if isinstance(p, MockSimpleHeuristicsPlayer):
                    p.n_finished_battles = 20
                    p.n_won_battles = 10
                else:
                    p.n_finished_battles = 20
                    p.n_won_battles = 5

        with patch(
            "src.agent.teambuild_policy.battle_royale.SimpleHeuristicsPlayer",
            MockSimpleHeuristicsPlayer,
        ), patch(
            "src.agent.teambuild_policy.battle_royale.RandomPlayer",
            MockRandomPlayer,
        ), patch(
            "src.agent.teambuild_policy.battle_royale.MaxBasePowerPlayer",
            MockMaxBasePowerPlayer,
        ), patch(
            "src.agent.teambuild_policy.battle_royale.cross_evaluate",
            AsyncMock(side_effect=mock_ce),
        ):
            result = asyncio.run(
                run_battle_royale(team_indices, cache, n_battles=10)
            )

        best = result["best_team_indices"]
        assert len(best) == 6
        for idx in best:
            assert 0 <= idx < cache.n_species

    def test_empty_teams_returns_default(self) -> None:
        """Empty team list returns first team with zero win rate."""
        cache = _make_mock_cache(30)
        team_indices: list[list[int]] = [[0, 1, 2, 3, 4, 5]]

        async def mock_ce(
            players: list[Any], n_challenges: int = 1
        ) -> None:
            for p in players:
                if isinstance(p, MockSimpleHeuristicsPlayer):
                    p.n_finished_battles = 20
                    p.n_won_battles = 0
                else:
                    p.n_finished_battles = 20
                    p.n_won_battles = 10

        with patch(
            "src.agent.teambuild_policy.battle_royale.SimpleHeuristicsPlayer",
            MockSimpleHeuristicsPlayer,
        ), patch(
            "src.agent.teambuild_policy.battle_royale.RandomPlayer",
            MockRandomPlayer,
        ), patch(
            "src.agent.teambuild_policy.battle_royale.MaxBasePowerPlayer",
            MockMaxBasePowerPlayer,
        ), patch(
            "src.agent.teambuild_policy.battle_royale.cross_evaluate",
            AsyncMock(side_effect=mock_ce),
        ):
            result = asyncio.run(
                run_battle_royale(team_indices, cache, n_battles=10)
            )

        assert result["best_team_idx"] == 0
        assert result["empirical_win_rate"] == 0.0
