import yaml
from pathlib import Path
from src.config.model import BattleWeights

def load_battle_weights(config_path: str = "src/config/battle_weights.yaml") -> BattleWeights:
    with open(Path(config_path), "r") as f:
        data = yaml.safe_load(f)
    return BattleWeights(**data)