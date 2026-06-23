import asyncio
import numpy as np

from src.agent.base import Pokemon, Move
from src.agent.selection_policy.double_oracle import (
    Strategy,
    find_best_response,
    enumerate_top_k_strategies,
)
from src.agent.selection_policy.optimizer import solve_sub_matrix, nash_loop


def _make_test_team() -> list[Pokemon]:
    return [
        Pokemon(
            species="snorlax",
            moves=[Move(name="bodyslam"), Move(name="earthquake")],
            nature="adamant", ev_hp=252, ev_atk=252,
        ) for _ in range(6)
    ]


def test_solve_sub_matrix_returns_nash_for_rock_paper_scissors() -> None:
    matrix = np.array([[0.5, 0.0, 1.0], [1.0, 0.5, 0.0], [0.0, 1.0, 0.5]], dtype=np.float64)
    mix = solve_sub_matrix(matrix)
    assert mix.shape == (3,)
    assert np.isclose(mix.sum(), 1.0, atol=0.01)
    assert np.all(mix >= -1e-10)
    expected = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)
    assert np.allclose(mix, expected, atol=0.01)


async def _mock_battle(
    our_team: list[Pokemon],
    our_strategy: Strategy,
    opponent_team: list[Pokemon],
    opponent_strategy: Strategy,
) -> float:
    our_score = float(our_strategy[0] + our_strategy[1])
    opp_score = float(opponent_strategy[0] + opponent_strategy[1])
    return 1.0 if our_score > opp_score else 0.0


def test_nash_loop_with_mock_battle_returns_distribution() -> None:
    team = _make_test_team()
    opp_team = _make_test_team()

    async def _run() -> None:
        mix, strategies = await nash_loop(
            team, opp_team,
            k_start=3, k_max=5, timeout_sec=10.0,
            battle_fn=_mock_battle,
        )
        assert len(mix) == 5
        assert len(strategies) > 0, "should return at least initial strategies"
        assert len(strategies) <= 5
        assert np.isclose(mix[: len(strategies)].sum(), 1.0, atol=0.01)

    asyncio.run(_run())


def test_expand_matrix_async_with_mock_battle() -> None:
    team = _make_test_team()
    opp_team = _make_test_team()

    our_strats: list[Strategy] = [
        (0, 1, 2, 3, 4, 5),
        (1, 0, 2, 3, 4, 5),
    ]
    opp_strats: list[Strategy] = [
        (2, 3, 0, 1, 4, 5),
        (3, 2, 0, 1, 4, 5),
    ]

    from src.agent.selection_policy.double_oracle import expand_matrix_async

    async def _run() -> None:
        matrix = await expand_matrix_async(
            our_strats, opp_strats,
            team, opp_team,
            battle_fn=_mock_battle,
        )
        assert matrix.shape == (2, 2)
        assert np.all((matrix >= 0) & (matrix <= 1))

    asyncio.run(_run())


def test_enumerate_with_opponent_team_affects_scoring() -> None:
    mixed_team = [
        Pokemon(species="snorlax", moves=[Move(name="bodyslam")],
                nature="adamant", ev_hp=252, ev_atk=252),
        Pokemon(species="snorlax", moves=[Move(name="bodyslam")],
                nature="adamant", ev_hp=252, ev_atk=252),
        Pokemon(species="snorlax", moves=[Move(name="earthquake")],
                nature="adamant", ev_hp=252, ev_atk=252),
        Pokemon(species="snorlax", moves=[Move(name="earthquake")],
                nature="adamant", ev_hp=252, ev_atk=252),
        Pokemon(species="snorlax", moves=[Move(name="bodyslam"), Move(name="earthquake")],
                nature="adamant", ev_hp=252, ev_atk=252),
        Pokemon(species="snorlax", moves=[Move(name="bodyslam"), Move(name="earthquake")],
                nature="adamant", ev_hp=252, ev_atk=252),
    ]
    fast_opp = [
        Pokemon(species="jolteon", nature="timid", ev_hp=4, ev_spa=252, ev_spe=252) for _ in range(6)
    ]
    slow_opp = [
        Pokemon(species="snorlax", nature="adamant", ev_hp=252, ev_atk=252) for _ in range(6)
    ]

    strats_default = enumerate_top_k_strategies(mixed_team, k=3, opponent_team=None)
    strats_fast = enumerate_top_k_strategies(mixed_team, k=3, opponent_team=fast_opp)
    strats_slow = enumerate_top_k_strategies(mixed_team, k=3, opponent_team=slow_opp)

    assert len(strats_default) == 3
    assert len(strats_fast) == 3
    assert len(strats_slow) == 3

    assert strats_fast != strats_slow, (
        "scoring should differ because opponent defense stats differ"
    )


def test_find_best_response_mock_matrix() -> None:
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    opp_mix = np.array([0.8, 0.2], dtype=np.float64)
    best = find_best_response(matrix, opp_mix)
    assert best == 0, "strategy 0 should be best against opponent favoring col 0"
