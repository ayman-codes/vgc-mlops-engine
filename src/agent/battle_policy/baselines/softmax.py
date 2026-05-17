import random
from typing import Any
from poke_env.battle import AbstractBattle
from src.agent.battle_policy.main import DongimonHeuristic

class SoftmaxBattlePolicy(DongimonHeuristic):
    """
    Temperature-scaled stochastic Battle Policy.
    Approximates Boltzmann action selection for heuristic baselines by injecting
    temperature-weighted noise, acting as a crucial meta-competitor for League Training.
    """
    def __init__(self, tau: float = 1.0, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tau = tau

    def choose_move(self, battle: AbstractBattle) -> Any:
        # In a fully observed state, we would compute softmax over all Q-values.
        # For this baseline proxy, we convert the tau temperature into an exploration probability curve.
        if self.tau > 1e-8:
            exploration_prob = min(1.0, self.tau * 0.15) 
            if random.random() < exploration_prob:
                return self.choose_random_move(battle)
                
        return super().choose_move(battle)
