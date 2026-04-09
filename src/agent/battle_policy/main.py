from typing import List, Tuple, Dict, Any
from vgc2.agent import BattlePolicy
from vgc2.battle_engine import BattleCommand
from vgc2.battle_engine.view import StateView
from vgc2.battle_engine.constants import BattleRuleParam

from src.config.loader import load_battle_weights
from src.config.model import BattleWeights
from src.agent.battle_policy.heuristics.scoring import _score_single_offensive_move, _score_protect_move, _score_single_switch_action
from src.agent.battle_policy.heuristics.threat import _identify_biggest_threat_opponent
from src.agent.battle_policy.heuristics.synergy import calculate_joint_synergy

class MyBattlePolicy(BattlePolicy): # type: ignore[misc]
    def __init__(self, detailed_logging: bool = False):
        super().__init__()
        self.weights: BattleWeights = load_battle_weights()
        self.detailed_logging: bool = detailed_logging
        self._telemetry_buffer: Dict[str, Any] = {}
        self.battle_params: BattleRuleParam = BattleRuleParam()

    def decision(self, state: StateView, turn_count: int) -> List[Tuple[int, int]]:
        my_team = state.sides[0].team.active
        active_indices = [i for i, p in enumerate(my_team) if p is not None and p.hp > 0]
        
        if not active_indices:
            return []
        
        if len(active_indices) == 1:
            # Fallback to standard single-unit evaluation
            return [self._evaluate_single_slot(state, active_indices[0])[0][0]]

        slot_0_idx = active_indices[0]
        slot_1_idx = active_indices[1]

        # 1. Independent Evaluation
        actions_A = self._evaluate_single_slot(state, slot_0_idx)
        actions_B = self._evaluate_single_slot(state, slot_1_idx)

        # 2. Truncation
        K = 7
        top_A = sorted(actions_A, key=lambda x: x[1], reverse=True)[:K]
        top_B = sorted(actions_B, key=lambda x: x[1], reverse=True)[:K]

        # 3. Synergistic Matrix Resolution
        best_joint_score = -float('inf')
        best_commands = ((0, 0), (0, 0))

        # Explicit unpacking
        threat_pkm, max_dmg, is_outsped, max_type_mult = _identify_biggest_threat_opponent(
            unit=my_team[slot_0_idx], 
            unit_side=0, 
            opponents=state.sides[1].team.active, 
            state=state,
            params=self.battle_params
            )

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

                if joint_q > best_joint_score:
                    best_joint_score = joint_q
                    best_commands = (cmd_A, cmd_B)
                    if self.detailed_logging:
                        cached_telemetry = (score_A, score_B, synergy_modifier)
        
        final_commands = list(best_commands)
        if self.detailed_logging:
            self._telemetry_buffer = {
                "turn_index": turn_count,
                "slot_0": {
                    "command": final_commands[0],
                    "raw_q_value": float(cached_telemetry[0]),
                    "synergy_contribution": float(cached_telemetry[2])
                },
                "slot_1": {
                    "command": final_commands[1],
                    "raw_q_value": float(cached_telemetry[1]),
                    "synergy_contribution": float(cached_telemetry[2])
                },
                "joint_q_score": float(best_joint_score)
            }

        return final_commands

    def get_telemetry(self) -> Dict[str, Any]:
        return self._telemetry_buffer

    def _evaluate_single_slot(self, state: StateView, slot_idx: int) -> List[Tuple[BattleCommand, float]]:
        actions = []
        unit = state.sides[0].team.active[slot_idx]
        
        if not unit or unit.hp <= 0:
            return [((0, 0), -float('inf'))]

        # Evaluate Moves
        if unit.battling_moves:
            for move_idx, move in enumerate(unit.battling_moves):
                if move.constants.protect:
                    score = _score_protect_move(unit, 0, state, self.battle_params)
                    actions.append(((move_idx, slot_idx), score))
                else:
                    opponents = state.sides[1].team.active
                    has_target = False
                    for target_idx, opp in enumerate(opponents):
                        if opp and opp.hp > 0:
                            has_target = True
                            score = _score_single_offensive_move(unit, opp, move, state, self.battle_params, 0)
                            actions.append(((move_idx, target_idx), score))
                    
                    if not has_target:
                        actions.append(((move_idx, 0), -100.0))

        # Evaluate Switches
        reserve = state.sides[0].team.reserve
        if reserve:
            for res_idx, res_pkm in enumerate(reserve):
                if res_pkm and res_pkm.hp > 0:
                    score = _score_single_switch_action(unit, res_pkm, 0, state, self.battle_params)
                    actions.append(((-1, res_idx), score))

        if not actions:
            actions.append(((0, 0), -float('inf')))

        return actions