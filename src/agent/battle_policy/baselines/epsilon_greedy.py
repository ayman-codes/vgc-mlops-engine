import random
from typing import Any
from poke_env.battle import AbstractBattle
from src.agent.battle_policy.main import DongimonHeuristic

class EpsilonGreedyBattlePolicy(DongimonHeuristic):
    """
    Epsilon-Greedy Battle Policy for PSRO exploration.
    Takes a random action with probability epsilon to prevent deterministic exploitation.
    """
    def __init__(self, epsilon: float = 0.2, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

    def choose_move(self, battle: AbstractBattle) -> Any:
        if random.random() < self.epsilon:
            return self.choose_random_move(battle)
        return super().choose_move(battle)
