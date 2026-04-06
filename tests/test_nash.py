import pytest
import numpy as np
from src.agent.selection_policy.inference.nash import solve_nash_equilibrium

def test_solve_nash_equilibrium_pure_strategy() -> None:
    matrix = np.array([
        [3.0, 1.0],
        [0.0, 0.0]
    ])
    strategy = solve_nash_equilibrium(matrix)
    
    assert strategy[0] > 0.99
    assert strategy[1] < 0.01
    assert pytest.approx(sum(strategy)) == 1.0

def test_solve_nash_equilibrium_mixed_strategy() -> None:
    matrix = np.array([
        [1.0, -1.0],
        [-1.0, 1.0]
    ])
    strategy = solve_nash_equilibrium(matrix)
    
    assert pytest.approx(strategy[0], abs=0.05) == 0.5
    assert pytest.approx(strategy[1], abs=0.05) == 0.5
    assert pytest.approx(sum(strategy)) == 1.0

def test_solve_nash_equilibrium_empty_matrix() -> None:
    matrix = np.zeros((0, 0), dtype=float)
    strategy = solve_nash_equilibrium(matrix)
    
    assert len(strategy) == 0

def test_solve_nash_equilibrium_single_row() -> None:
    matrix = np.array([[0.8, 0.4]])
    strategy = solve_nash_equilibrium(matrix)
    
    assert len(strategy) == 1
    assert strategy[0] == 1.0

if __name__ == "__main__":
    pytest.main([__file__])