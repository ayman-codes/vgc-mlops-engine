from typing import Any, List, Tuple, Union
from poke_env.player import SimpleHeuristicsPlayer
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player.battle_order import DoubleBattleOrder, SingleBattleOrder, DefaultBattleOrder

from src.agent.battle_policy.heuristics.scoring import _score_single_offensive_move, _score_protect_move, _score_single_switch_action
from src.agent.battle_policy.heuristics.threat import _identify_biggest_threat_opponent
from src.agent.battle_policy.heuristics.synergy import calculate_joint_synergy, DummyWeights

class DongimonHeuristic(SimpleHeuristicsPlayer):
    """
    Restored and translated Custom Greedy Battle Policy.
    Evaluates individual moves and calculates a Joint Synergy Matrix.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.weights = DummyWeights()

    def choose_move(self, battle: AbstractBattle) -> Any:
        if not isinstance(battle, DoubleBattle):
            return super().choose_move(battle)

        if battle.in_team_preview:
            return super().choose_move(battle)

        if battle.finished:
            return DefaultBattleOrder()  # type: ignore[no-untyped-call]

        active = battle.active_pokemon
        if not active[0] and not active[1]:
            return super().choose_move(battle)

        pkm_a = active[0] if active[0] else active[1]
        pkm_b = active[1] if active[1] else active[0]
        if not pkm_a or not pkm_b:
            return super().choose_move(battle)

        actions_A = self._evaluate_single_slot(battle, 0)
        actions_B = self._evaluate_single_slot(battle, 1)

        K = 5
        top_A = sorted(actions_A, key=lambda x: x[1], reverse=True)[:K]
        top_B = sorted(actions_B, key=lambda x: x[1], reverse=True)[:K]

        opponents = [opp for opp in battle.opponent_active_pokemon if opp is not None and not opp.fainted]
        
        threat_pkm, _, _, _ = _identify_biggest_threat_opponent(
            unit=pkm_a,
            unit_side=0,
            opponents=opponents,
            state=battle,
            params=None
        )

        best_joint_score = -float('inf')
        best_commands: Tuple[Any, Any] = (None, None)

        if not top_A and not top_B:
            return super().choose_move(battle)

        if not top_A:
            best_cmd_B = self._format_order(top_B[0][0]) if top_B else DefaultBattleOrder()  # type: ignore[no-untyped-call]
            return DoubleBattleOrder(DefaultBattleOrder(), best_cmd_B)  # type: ignore[no-untyped-call]
            
        if not top_B:
            best_cmd_A = self._format_order(top_A[0][0]) if top_A else DefaultBattleOrder()  # type: ignore[no-untyped-call]
            return DoubleBattleOrder(best_cmd_A, DefaultBattleOrder())  # type: ignore[no-untyped-call]

        for cmd_A, score_A in top_A:
            for cmd_B, score_B in top_B:
                synergy = calculate_joint_synergy(
                    state=battle,
                    cmd_A=cmd_A,
                    cmd_B=cmd_B,
                    unit_A=pkm_a,
                    unit_B=pkm_b,
                    biggest_threat=threat_pkm,
                    weights=self.weights,
                    score_A=score_A,
                    score_B=score_B
                )
                
                joint_q = (score_A * self.weights.W_BASE_SCORE_A) + (score_B * self.weights.W_BASE_SCORE_B) + synergy
                
                if joint_q > best_joint_score:
                    best_joint_score = joint_q
                    best_commands = (cmd_A, cmd_B)
                    
        cmd1 = self._format_order(best_commands[0])
        cmd2 = self._format_order(best_commands[1])
        
        try:
            return DoubleBattleOrder(cmd1, cmd2)
        except Exception:
            return super().choose_move(battle)
            
    def _format_order(self, cmd: Any) -> Union[SingleBattleOrder, DefaultBattleOrder]:
        if isinstance(cmd, tuple):
            move_or_switch, target = cmd
            if target is None:
                return SingleBattleOrder(move_or_switch)
            return SingleBattleOrder(move_or_switch, move_target=target)
        return DefaultBattleOrder()  # type: ignore[no-untyped-call]

    def _evaluate_single_slot(self, battle: DoubleBattle, slot_idx: int) -> List[Tuple[Any, float]]:
        actions: List[Tuple[Any, float]] = []
        unit = battle.active_pokemon[slot_idx]
        
        if not unit or unit.fainted:
            return actions

        if battle.force_switch:
            if battle.available_switches[slot_idx]:
                for res_pkm in battle.available_switches[slot_idx]:
                    if res_pkm and not res_pkm.fainted:
                        score = _score_single_switch_action(unit, res_pkm, slot_idx, battle, None)
                        actions.append(((res_pkm, None), score))
            return actions

        if battle.available_moves[slot_idx]:
            for move in battle.available_moves[slot_idx]:
                if move.id == 'protect':
                    score = _score_protect_move(unit, slot_idx, battle, None)
                    actions.append(((move, 1), score))
                else:
                    opponents = battle.opponent_active_pokemon
                    has_target = False
                    for t_idx, opp in enumerate(opponents):
                        if opp and not opp.fainted:
                            has_target = True
                            actual_target = t_idx + 1
                            score = _score_single_offensive_move(unit, opp, move, battle, None, actual_target)
                            actions.append(((move, actual_target), score))
                    if not has_target:
                        actions.append(((move, 1), -10.0))

        if battle.available_switches[slot_idx]:
            for res_pkm in battle.available_switches[slot_idx]:
                if res_pkm and not res_pkm.fainted:
                    score = _score_single_switch_action(unit, res_pkm, slot_idx, battle, None)
                    actions.append(((res_pkm, None), score))
                
        return actions
