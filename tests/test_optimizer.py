import numpy as np

from src.agent.selection_policy.optimizer import solve_sub_matrix


def test_solve_sub_matrix_valid_nash_equilibrium() -> None:
    matrix = np.array([[0.6, 0.4], [0.3, 0.7]], dtype=np.float64)
    mix = solve_sub_matrix(matrix)
    assert mix.shape == (2,)
    assert np.isclose(mix.sum(), 1.0, atol=0.01)
    val = mix @ matrix
    assert np.all(val >= val.min() - 1e-6)


def test_solve_sub_matrix_pure_for_dominant_row() -> None:
    matrix = np.array([[0.8, 0.9], [0.2, 0.1]], dtype=np.float64)
    mix = solve_sub_matrix(matrix)
    assert mix[0] > 0.9


def test_solve_sub_matrix_sums_to_one() -> None:
    matrix = np.array([[0.6, 0.4], [0.3, 0.7]], dtype=np.float64)
    mix = solve_sub_matrix(matrix)
    assert np.isclose(mix.sum(), 1.0, atol=0.01)


def test_solve_sub_matrix_all_nonnegative() -> None:
    matrix = np.array([[0.6, 0.4], [0.3, 0.7], [0.5, 0.5]], dtype=np.float64)
    mix = solve_sub_matrix(matrix)
    assert np.all(mix >= -1e-10)


def test_solve_sub_matrix_single_strategy() -> None:
    matrix = np.array([[0.5]], dtype=np.float64)
    mix = solve_sub_matrix(matrix)
    assert len(mix) == 1
    assert np.isclose(mix[0], 1.0, atol=0.01)


def test_solve_sub_matrix_3x3() -> None:
    matrix = np.array([[0.5, 0.6, 0.4], [0.4, 0.3, 0.7], [0.6, 0.5, 0.3]], dtype=np.float64)
    mix = solve_sub_matrix(matrix)
    assert mix.shape == (3,)
    assert np.isclose(mix.sum(), 1.0, atol=0.01)
    assert np.all(mix >= -1e-10)
