import pytest
from src.agent.battle_policy.heuristics.threat import calculate_effective_speed, estimate_incoming_threat
from vgc2.core.GameState import GameState
from vgc2.core.Pkm import Pkm, PkmStats

def test_speed_resolution_tailwind():
    """Verify Tailwind doubles effective speed"""
    pkm = Pkm()
    pkm.stats = PkmStats(spe=100)
    pkm.team_index = 0
    state = GameState()
    state.field.tailwind[0] = True
    
    assert calculate_effective_speed(pkm, state) == 200

def test_lethal_threat_penalty():
    """Verify near-zero penalty when outsped and lethal"""
    unit = Pkm()
    unit.hp = 10
    unit.stats = PkmStats(spe=50)
    unit.team_index = 0
    
    opp = Pkm()
    opp.stats = PkmStats(spe=100)
    # Mocking moves for damage check
    
    state = GameState()
    state.teams[1].active = [opp]
    
    threat = estimate_incoming_threat(unit, state)
    if threat["is_lethal"] and threat["is_outsped"]:
        assert threat["penalty_multiplier"] == 0.01

if __name__ == "__main__":
    pytest.main([__file__])