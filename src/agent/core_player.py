from typing import Any
from poke_env.player import Player
from poke_env.battle import AbstractBattle


class MyVGCAgent(Player):
    """
    Core VGC Agent combining Battle Policy and Selection Policy via poke-env.
    """
    
    def choose_move(self, battle: AbstractBattle) -> Any:
        """
        Battle Policy entrypoint.
        """
        return self.choose_random_move(battle)

    def teampreview(self, battle: AbstractBattle) -> str:
        """
        Selection Policy (Team Preview) entrypoint.
        """
        return "/team 1234"
