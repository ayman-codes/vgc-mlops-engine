from pydantic import BaseModel, Field


class BattleWeights(BaseModel):
    W_OFF_DEF_SUPPORT_BONUS: float = 0.0200
    W_BASE_SCORE_A: float = 0.0500
    W_ENV_SYNERGY_BONUS: float = 0.0200
    W_FOCUS_FIRE_BONUS: float = 0.2700
    W_TARGET_PRIORITY_BONUS: float = 0.1800
    W_BASE_SCORE_B: float = 0.1500
    W_SURVIVAL_IMPACT: float = 0.1300
    W_SETUP_SYNERGY_BONUS: float = 0.1800


class SelectionHeuristicsConfig(BaseModel):
    mixed_stat_threshold: float = 15.0
    protect_turn_value: float = 1.0
    toxic_damage_coefficient: float = 0.625
    paralysis_denial_chance: float = 0.25
    sleep_denial_turns: float = 1.5
    status_score_weight: float = 1.0
    weather_score_weight: float = 1.0
    terrain_score_weight: float = 1.0
    base_hp_reference_turns: float = 16.0


class SelectionConfig(BaseModel):
    heuristics: SelectionHeuristicsConfig = Field(default_factory=SelectionHeuristicsConfig)
    simulation_timeout_sec: float = 85.0
    min_simulations_per_matchup: int = 1
    use_nash_equilibrium: bool = False
    procedural_variance: float = 0.20
    timeout_limit_sec: float = 60.0
    async_batch_size: int = 4


class AppConfig(BaseModel):
    mlflow_tracking_uri: str = "http://localhost:5000"
    showdown_server_url: str = "http://localhost:8000"
    data_root: str = "data"
    model_root: str = "models"
