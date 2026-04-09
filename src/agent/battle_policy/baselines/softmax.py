import numpy as np
from typing import List, Tuple, Sequence, TypeVar
from vgc2.battle_engine.view import StateView
from src.agent.battle_policy.main import MyBattlePolicy
from src.agent.battle_policy.heuristics.threat import _identify_biggest_threat_opponent
from src.agent.battle_policy.heuristics.synergy import calculate_joint_synergy

T = TypeVar('T')

class SoftmaxBattlePolicy(MyBattlePolicy):
    def __init__(self, tau: float = 1.0) -> None:
        super().__init__()
        self.tau = tau

    def decision(self, state: StateView, turn_count: int) -> List[Tuple[int, int]]:
        my_team = state.sides[0].team.active
        active_indices = [i for i, p in enumerate(my_team) if p is not None and p.hp > 0]

        if not active_indices:
            return []

        if len(active_indices) == 1:
            actions = self._evaluate_single_slot(state, active_indices[0])
            if not actions:
                return [(0, 0)]
            cmds, scores = zip(*actions)
            chosen_cmd = self._sample_softmax(cmds, scores)
            return [chosen_cmd]

        slot_0_idx = active_indices[0]
        slot_1_idx = active_indices[1]

        actions_A = self._evaluate_single_slot(state, slot_0_idx)
        actions_B = self._evaluate_single_slot(state, slot_1_idx)

        K = 5
        top_A = sorted(actions_A, key=lambda x: x[1], reverse=True)[:K]
        top_B = sorted(actions_B, key=lambda x: x[1], reverse=True)[:K]

        threat_pkm, _, _, _ = _identify_biggest_threat_opponent(
            unit=my_team[slot_0_idx],
            unit_side=0,
            opponents=state.sides[1].team.active,
            state=state,
            params=self.battle_params
        )

        joint_actions: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        joint_scores: List[float] = []

        for cmd_A, score_A in top_A:
            for cmd_B, score_B in top_B:
                synergy_modifier = calculate_joint_synergy(
                    state=state,
                    cmd_A=cmd_A,
                    cmd_B=cmd_B,
                    unit_A=my_team[slot_0_idx],
                    unit_B=my_team[slot_1_idx],
                    biggest_threat=threat_pkm,
                    weights=self.weights,
                    score_A=score_A,
                    score_B=score_B
                )

                joint_q = (score_A * self.weights.W_BASE_SCORE_A) + \
                          (score_B * self.weights.W_BASE_SCORE_B) + \
                          synergy_modifier

                joint_actions.append((cmd_A, cmd_B))
                joint_scores.append(float(joint_q))

        if not joint_actions:
            return [(0, 0), (0, 0)]

        selected_joint_action = self._sample_softmax(joint_actions, joint_scores)
        return list(selected_joint_action)

    def _sample_softmax(self, actions: Sequence[T], scores: Sequence[float]) -> T:
        scores_array = np.array(scores, dtype=np.float64)
        max_score = np.max(scores_array)
        
        scaled_scores = (scores_array - max_score) / self.tau
        exp_scores = np.exp(scaled_scores)
        probabilities = exp_scores / np.sum(exp_scores)
        
        chosen_idx = int(np.random.choice(len(actions), p=probabilities))
        return actions[chosen_idx]