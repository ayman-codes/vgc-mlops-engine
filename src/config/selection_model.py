from __future__ import annotations
from pydantic import BaseModel, Field, field_validator

class SelectionHeuristicsConfig(BaseModel):
    mixed_stat_threshold: float = Field(default=15.0, ge=0)
    protect_turn_value: float = Field(default=1.0, ge=0)
    toxic_damage_coefficient: float = Field(default=10.0 / 16.0, ge=0)
    paralysis_denial_chance: float = Field(default=0.25, ge=0, le=1)
    sleep_denial_turns: float = Field(default=1.5, gt=0)
    status_score_weight: float = Field(default=1.0, gt=0)
    weather_score_weight: float = Field(default=1.0, gt=0)
    terrain_score_weight: float = Field(default=1.0, gt=0)
    base_hp_reference_turns: float = Field(default=16.0, gt=0)

    @field_validator("protect_turn_value", "toxic_damage_coefficient")
    @classmethod
    def validate_bounds(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Score coefficients must be non-negative")
        return v

class SelectionConfig(BaseModel):
    heuristics: SelectionHeuristicsConfig = Field(default_factory=SelectionHeuristicsConfig)
    simulation_timeout_sec: float = Field(default=85.0, gt=0)
    min_simulations_per_matchup: int = Field(default=1, gt=0)
    use_nash_equilibrium: bool = Field(default=False)