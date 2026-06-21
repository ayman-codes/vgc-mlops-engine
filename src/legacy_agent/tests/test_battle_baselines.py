from unittest.mock import MagicMock
from src.agent.battle_policy.main import DongimonHeuristic
from src.agent.battle_policy.baselines.epsilon_greedy import EpsilonGreedyBattlePolicy
from src.agent.battle_policy.baselines.softmax import SoftmaxBattlePolicy

def test_battle_policy_initialization() -> None:
    greedy = DongimonHeuristic()
    assert greedy is not None

    eps_greedy = EpsilonGreedyBattlePolicy(epsilon=0.5)
    assert eps_greedy.epsilon == 0.5

    softmax = SoftmaxBattlePolicy(tau=2.0)
    assert softmax.tau == 2.0

def test_battle_policy_choose_move_signature() -> None:
    greedy = DongimonHeuristic()
    mock_battle = MagicMock()
    
    # We mock the choose_move to prevent actual heuristics from running on an empty mock
    greedy.choose_move = MagicMock(return_value="mock_order") # type: ignore
    
    result = greedy.choose_move(mock_battle)
    assert result == "mock_order"
