import numpy as np
from typing import Any, cast
from scipy.optimize import linprog

def solve_nash_equilibrium(payoff_matrix: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    num_rows, num_cols = payoff_matrix.shape
    if num_rows == 0 or num_cols == 0:
        return np.array([], dtype=np.float64)
        
    if num_rows == 1:
        return np.array([1.0], dtype=np.float64)

    min_val = float(np.min(payoff_matrix))
    offset = 0.0
    if min_val <= 0:
        offset = -min_val + 1.0
    
    adjusted_matrix = payoff_matrix + offset

    c = np.ones(num_rows)
    a_ub = -adjusted_matrix.T
    b_ub = -np.ones(num_cols)

    res = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=(0, None), method='highs')

    if not res.success:
        return np.ones(num_rows, dtype=np.float64) / float(num_rows)

    denominator = float(np.sum(res.x))
    strategy = np.array(res.x, dtype=np.float64) / denominator
    return cast(np.ndarray[Any, Any], strategy)