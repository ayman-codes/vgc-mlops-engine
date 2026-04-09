import random
from typing import Optional, List, cast

from vgc2.agent.battle import GreedyBattlePolicy # type: ignore
from vgc2.battle_engine.game_state import State
from vgc2.battle_engine.view import TeamView
from vgc2.battle_engine import BattleCommand

class EpsilonGreedyBattlePolicy(GreedyBattlePolicy):
    def __init__(self, epsilon: float = 0.2):
        super().__init__()
        self.epsilon = epsilon

    def decision(self,
                 state: State,
                 opp_view: Optional[TeamView] = None) -> List[BattleCommand]:
        
        if random.random() >= self.epsilon:
            return cast(List[BattleCommand], super().decision(state, opp_view))
        
        cmds: List[BattleCommand] = []
        my_team = state.sides[0].team
        opp_active = state.sides[1].team.active

        valid_targets = [i for i, opp in enumerate(opp_active) if opp is not None and opp.hp > 0]
        if not valid_targets:
            valid_targets = [0]

        for pkm in my_team.active:
            if not pkm or pkm.hp <= 0:
                cmds.append((0, 0))
                continue

            valid_actions: List[BattleCommand] = []

            for m_idx, move in enumerate(pkm.battling_moves):
                if move.pp > 0:
                    for t_idx in valid_targets:
                        valid_actions.append((m_idx, t_idx))

            for r_idx, r_pkm in enumerate(my_team.reserve):
                if r_pkm is not None and r_pkm.hp > 0:
                    valid_actions.append((-(r_idx + 1), 0))

            if not valid_actions:
                cmds.append((0, 0))
            else:
                cmds.append(random.choice(valid_actions))

        return cmds