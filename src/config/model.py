from pydantic import BaseModel, Field

class BattleWeights(BaseModel):
    W_OFF_DEF_SUPPORT_BONUS: float = Field(ge=0.0, le=1.0)
    W_BASE_SCORE_A: float = Field(ge=0.0, le=1.0)
    W_ENV_SYNERGY_BONUS: float = Field(ge=0.0, le=1.0)
    W_FOCUS_FIRE_BONUS: float = Field(ge=0.0, le=1.0)
    W_TARGET_PRIORITY_BONUS: float = Field(ge=0.0, le=1.0)
    W_BASE_SCORE_B: float = Field(ge=0.0, le=1.0)
    W_SURVIVAL_IMPACT: float = Field(ge=0.0, le=1.0)
    W_SETUP_SYNERGY_BONUS: float = Field(ge=0.0, le=1.0)