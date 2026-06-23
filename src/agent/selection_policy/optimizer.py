"""Nash equilibrium solver via linear programming and Double Oracle loop.

Solves zero-sum matrix games using scipy.optimize.linprog and
iterates strategy sets via the Double Oracle algorithm with MLflow telemetry.
"""

import time
from typing import Any
import numpy as np
from numpy.typing import NDArray
import mlflow

from src.agent.selection_policy.double_oracle import (
    Strategy,
    enumerate_top_k_strategies,
    expand_matrix_async,
    _score_lead_pair,
    _pokemon_stats,
    _species_types,
)


def solve_sub_matrix(payoff_matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute the Nash equilibrium mixture for a zero-sum payoff matrix.

    Solves the LP: max_v s.t. x @ payoff >= v, sum(x) = 1, x >= 0.

    Args:
        payoff_matrix: Shape (our_strategies, opponent_strategies) — our win rates.

    Returns:
        Probability vector of length our_strategies.
    """
    from scipy.optimize import linprog

    n, m = payoff_matrix.shape
    c = np.zeros(n + 1)
    c[-1] = -1.0

    A_ub = np.zeros((m, n + 1))
    A_ub[:, :n] = -payoff_matrix.T
    A_ub[:, n] = 1.0
    b_ub = np.zeros(m)

    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0.0, 1.0)] * n + [(None, None)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if not result.success:
        return np.ones(n, dtype=np.float64) / n

    x_arr = np.asarray(result.x[:n], dtype=np.float64)
    norm: NDArray[np.float64] = x_arr / x_arr.sum()
    return norm


async def nash_loop(
    our_team: list[Any],
    opponent_team: list[Any],
    oracle_runner: Any = None,
    k_start: int = 3,
    k_max: int = 15,
    timeout_sec: float = 60.0,
    battle_fn: Any = None,
) -> tuple[NDArray[np.float64], list[Strategy]]:
    """Run the Double Oracle loop and return the Nash mixture over our strategies.

    Args:
        our_team: Our 6 Pokemon.
        opponent_team: Opponent's 6 Pokemon.
        oracle_runner: Async callable for matrix expansion (defaults to expand_matrix_async).
        k_start: Initial number of strategies per side.
        k_max: Maximum number of strategies per side after expansion.
        timeout_sec: Wall-clock timeout for the loop.
        battle_fn: Battle function passed to expand_matrix_async.

    Returns:
        Tuple of (probability vector padded to k_max, final strategy list)
        where position i in the vector corresponds to strategy i in the list.
    """
    our_strategies = enumerate_top_k_strategies(our_team, k=k_start, opponent_team=opponent_team)
    opp_strategies = enumerate_top_k_strategies(opponent_team, k=k_start, opponent_team=our_team)

    if not our_strategies or not opp_strategies:
        return np.ones(k_max, dtype=np.float64) / max(k_max, 1), []

    runner = oracle_runner or expand_matrix_async
    start_time = time.time()

    with mlflow.start_run(nested=True, run_name="nash-loop"):
        mlflow.log_params({
            "k_start": k_start,
            "k_max": k_max,
            "timeout_sec": timeout_sec,
        })

        iteration = 0
        added = True

        while added and len(our_strategies) < k_max and len(opp_strategies) < k_max:
            elapsed = time.time() - start_time
            if elapsed > timeout_sec:
                mlflow.log_metric("timeout_hit", 1.0, step=iteration)
                break

            iteration += 1
            mlflow.log_metric("iteration", iteration, step=iteration)
            mlflow.log_metric("our_strategies", len(our_strategies), step=iteration)
            mlflow.log_metric("opp_strategies", len(opp_strategies), step=iteration)

            matrix = await runner(
                our_strategies, opp_strategies,
                our_team, opponent_team,
                battle_fn=battle_fn,
            )
            mlflow.log_metric("payoff_mean", float(matrix.mean()), step=iteration)

            our_mix = solve_sub_matrix(matrix)
            opp_mix = solve_sub_matrix(matrix.T)

            our_best = _next_strategy(
                our_strategies, our_team, opp_mix, opponent_team, opp_strategies, k_max,
            )
            opp_best = _next_strategy(
                opp_strategies, opponent_team, our_mix, our_team, our_strategies, k_max,
            )

            if our_best is None and opp_best is None:
                added = False
            else:
                if our_best is not None:
                    our_strategies.append(our_best)
                if opp_best is not None:
                    opp_strategies.append(opp_best)

        matrix = await runner(
            our_strategies, opp_strategies,
            our_team, opponent_team,
            battle_fn=battle_fn,
        )
        final_mix = solve_sub_matrix(matrix)

        mlflow.log_metric("final_value", float((matrix @ final_mix).mean()), step=iteration)
        mlflow.log_metric("total_iterations", iteration)

    padded = np.zeros(k_max, dtype=np.float64)
    padded[: len(final_mix)] = final_mix
    return padded, our_strategies


def _next_strategy(
    current: list[Strategy],
    team: list[Any],
    opponent_mix: NDArray[np.float64] | None,
    opponent_team: list[Any],
    opp_strategies: list[Strategy],
    k_max: int,
) -> Strategy | None:
    if len(current) >= k_max:
        return None
    if opponent_mix is None or len(opponent_mix) == 0:
        return None
    if not opp_strategies:
        return None

    existing = set(current)
    candidates: list[Strategy] = []
    for k in range(len(current) + 1, k_max + 1):
        for s in enumerate_top_k_strategies(team, k=k, opponent_team=opponent_team):
            if s not in existing:
                candidates.append(s)
                existing.add(s)
    if not candidates:
        return None

    best_score = -float("inf")
    best_cand: Strategy | None = None

    for cand in candidates:
        cand_lead1 = team[cand[0]]
        cand_lead2 = team[cand[1]]
        expected = 0.0
        for j, opp_strat in enumerate(opp_strategies):
            w = opponent_mix[j]
            if w <= 0:
                continue
            opp_lead1 = opponent_team[opp_strat[0]]
            opp_lead2 = opponent_team[opp_strat[1]]
            opp_types = list(
                set(_species_types(opp_lead1.species) + _species_types(opp_lead2.species))
            )
            o_stats1 = _pokemon_stats(opp_lead1)
            o_stats2 = _pokemon_stats(opp_lead2)
            avg_def = (o_stats1["def"] + o_stats2["def"]) / 2
            avg_spd = (o_stats1["spd"] + o_stats2["spd"]) / 2
            score = _score_lead_pair(cand_lead1, cand_lead2, opp_types, avg_def, avg_spd)
            expected += w * score
        if expected > best_score:
            best_score = expected
            best_cand = cand

    return best_cand
