import pytest
from unittest.mock import MagicMock, patch
from src.agent.selection_policy.inference.payoff import generate_payoff_matrix

@patch("src.agent.selection_policy.inference.payoff.run_sub_tournament", return_value=0.75)
def test_generate_payoff_matrix_dimensions(mock_run) -> None:
    my_team = MagicMock()
    
    my_pairs = [(0, 1), (1, 2), (2, 3)]
    opp_pairs = [(MagicMock(), MagicMock()), (MagicMock(), MagicMock())]
    
    predicted_dict = {}
    sim_policy = MagicMock()
    
    matrix = generate_payoff_matrix(my_team, my_pairs, opp_pairs, predicted_dict, sim_policy)
    
    assert matrix.shape == (3, 2)
    assert matrix[0, 0] == 0.75
    assert mock_run.call_count == 6

def test_generate_payoff_matrix_empty_input() -> None:
    my_team = MagicMock()
    matrix = generate_payoff_matrix(my_team, [], [], {}, MagicMock())
    
    assert matrix.shape == (0, 0)

if __name__ == "__main__":
    pytest.main([__file__])