import numpy as np
from hypothesis import given
import hypothesis.strategies as st
from src.agent.selection_policy.bayesian_do_selection import BayesianDoubleOraclePolicy

def test_nash_equilibrium_resolution_validity() -> None:
    policy = BayesianDoubleOraclePolicy()
    payoff = np.array([
        [1.0, -1.0],
        [-1.0, 1.0]
    ])
    strategy = policy.resolve_nash_equilibrium(payoff)
    
    assert len(strategy) == 2
    assert np.isclose(np.sum(strategy), 1.0)
    assert np.all(strategy >= 0.0)

@given(st.lists(st.lists(st.floats(min_value=-10.0, max_value=10.0), min_size=15, max_size=15), min_size=15, max_size=15))
def test_nash_equilibrium_property_15x15(matrix: list[list[float]]) -> None:
    payoff = np.array(matrix)
    policy = BayesianDoubleOraclePolicy()
    strategy = policy.resolve_nash_equilibrium(payoff)
    
    assert len(strategy) == 15
    assert np.isclose(np.sum(strategy), 1.0)
    assert np.all(strategy >= -1e-8) # floating point tolerance for bounds
