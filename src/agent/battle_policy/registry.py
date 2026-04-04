from typing import Any, Dict
from src.agent.battle_policy.main import MyBattlePolicy
from vgc2.agent import BattlePolicy

def get_policy(config: Dict[str, Any] | None = None) -> BattlePolicy:
    """VGC2 tournament runner entry point.
    Expects optional config dict from track orchestrator.
    """
    if config is None:
        config = {}
    detailed_logging = bool(config.get("detailed_logging", False))
    return MyBattlePolicy(detailed_logging=detailed_logging)