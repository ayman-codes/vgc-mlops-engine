from typing import Any, Dict
from src.agent.battle_policy.main import DongimonHeuristic

def get_policy(config: Dict[str, Any] | None = None) -> DongimonHeuristic:
    """Entry point for battle policy creation.
    Expects optional config dict from track orchestrator.
    """
    if config is None:
        config = {}
    detailed_logging = bool(config.get("detailed_logging", False))
    return DongimonHeuristic(detailed_logging=detailed_logging)