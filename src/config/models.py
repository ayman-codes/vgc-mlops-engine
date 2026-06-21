from pydantic import BaseModel


class BattleWeights(BaseModel):
    W_OFF_DEF_SUPPORT_BONUS: float = 0.0200
    W_BASE_SCORE_A: float = 0.0500
    W_ENV_SYNERGY_BONUS: float = 0.0200
    W_FOCUS_FIRE_BONUS: float = 0.2700
    W_TARGET_PRIORITY_BONUS: float = 0.1800
    W_BASE_SCORE_B: float = 0.1500
    W_SURVIVAL_IMPACT: float = 0.1300
    W_SETUP_SYNERGY_BONUS: float = 0.1800



class AppConfig(BaseModel):
    mlflow_tracking_uri: str = "http://localhost:5000"
    showdown_server_url: str = "http://localhost:8000"
    data_root: str = "data"
    model_root: str = "models"
