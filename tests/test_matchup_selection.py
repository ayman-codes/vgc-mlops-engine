import pytest
from unittest.mock import MagicMock, patch
from src.agent.selection_policy.heuristics.matchup import generate_team_combinations, run_sub_tournament

def test_generate_team_combinations() -> None:
    source_team = MagicMock()
    source_team.members = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
    combinations = generate_team_combinations(source_team, 2)
    assert len(combinations) == 6
    assert (0, 1) in combinations
    assert (2, 3) in combinations

def test_generate_team_combinations_invalid() -> None:
    source_team = MagicMock()
    source_team.members = [MagicMock()]
    combinations = generate_team_combinations(source_team, 2)
    assert len(combinations) == 0

@patch("src.agent.selection_policy.heuristics.matchup.BattleEngine")
@patch("src.agent.selection_policy.heuristics.matchup.State")
@patch("src.agent.selection_policy.heuristics.matchup.BattlingTeam")
def test_run_sub_tournament(mock_battling_team: MagicMock, mock_state: MagicMock, mock_engine_cls: MagicMock) -> None:
    mock_engine = MagicMock()
    mock_engine.finished.side_effect = [False, True]
    mock_engine.winning_side = 0
    mock_engine_cls.return_value = mock_engine

    my_team = MagicMock()
    my_team.members = [MagicMock(), MagicMock(), MagicMock()]
    
    view_1 = MagicMock()
    view_2 = MagicMock()
    
    build_a = [MagicMock()]
    build_b = [MagicMock()]
    
    predicted_dict = {view_1: build_a, view_2: build_b}
    
    sim_policy = MagicMock()
    sim_policy.decision.return_value = (0, 0)
    
    win_rate = run_sub_tournament(my_team, (0, 1), (view_1, view_2), predicted_dict, sim_policy)
    
    assert win_rate == 1.0

def test_run_sub_tournament_missing_builds() -> None:
    my_team = MagicMock()
    view_1 = MagicMock()
    view_2 = MagicMock()
    
    predicted_dict = {view_1: [MagicMock()]} 
    sim_policy = MagicMock()
    
    win_rate = run_sub_tournament(my_team, (0, 1), (view_1, view_2), predicted_dict, sim_policy)
    
    assert win_rate == 0.0

if __name__ == "__main__":
    pytest.main([__file__])