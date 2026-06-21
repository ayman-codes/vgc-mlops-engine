from typing import Any

import gymnasium as gym

from src.config.models import BattleWeights


class HeuristicRewardWrapper(gym.Wrapper[Any, Any, Any, Any]):
    def __init__(self, env: gym.Env[Any, Any], weights: BattleWeights | None = None):
        super().__init__(env)
        self.weights = weights or BattleWeights()

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        heuristic_bonus = self._compute_heuristic_score(obs, action, info)
        reward = float(reward) + heuristic_bonus
        return obs, reward, terminated, truncated, info

    def _compute_heuristic_score(self, obs: Any, action: Any, info: dict[str, Any]) -> float:
        damage_dealt = float(info.get("damage_dealt", 0))
        ko_scored = bool(info.get("ko_scored", False))
        fainted = bool(info.get("fainted", False))
        switch_out = bool(info.get("switched", False))

        score = 0.0
        score += damage_dealt * self.weights.W_FOCUS_FIRE_BONUS
        if ko_scored:
            score += self.weights.W_TARGET_PRIORITY_BONUS
        if fainted:
            score -= self.weights.W_SURVIVAL_IMPACT
        if switch_out:
            score += self.weights.W_OFF_DEF_SUPPORT_BONUS * 0.5
        return score
