import yaml
from pathlib import Path
from src.config.models import BattleWeights
from src.config.selection_model import SelectionConfig


def load_battle_weights(config_path: str = "src/config/battle_weights.yaml") -> BattleWeights:
    with open(Path(config_path), "r") as f:
        data = yaml.safe_load(f)
    return BattleWeights(**data)


def load_selection_config(config_path: str = "src/config/selection_weights.yaml") -> SelectionConfig:
    with open(Path(config_path), "r") as f:
        data = yaml.safe_load(f)
    return SelectionConfig(**{k: v for k, v in data.items() if k in SelectionConfig.model_fields})