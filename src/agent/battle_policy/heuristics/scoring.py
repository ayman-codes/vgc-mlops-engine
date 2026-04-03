from typing import List
from vgc2.core.GameState import GameState
from vgc2.core.Pkm import Pkm, Move
from vgc2.battle_engine.damage_calculator import calculate_damage
from .threat import estimate_incoming_threat

MAX_SCORE = 1000.0

def _score_single_offensive_move(attacker: Pkm, target: Pkm, move: Move, state: GameState) -> float:
    if move.pp <= 0 or target.hp <= 0:
        return -float('inf')

    damage = calculate_damage(attacker, target, move, state)
    
    # Calculate base percentage of HP removed
    score = (damage / target.max_hp) * 100.0
    
    # KO Bonus: Prioritize moves that secure a knockout
    if damage >= target.hp:
        score += 500.0
        
    return min(score, MAX_SCORE)

def _score_protect_move(unit: Pkm, state: GameState) -> float:
    threat = estimate_incoming_threat(unit, state)
    
    # Scale score based on lethal threat and outspeed status
    if threat["is_lethal"] and threat["is_outsped"]:
        return 600.0
    
    # Moderate score for general damage mitigation
    return threat["aggro_score"] * 300.0

def _score_single_switch_action(unit: Pkm, switch_in: Pkm, state: GameState) -> float:
    if switch_in.hp <= 0:
        return -float('inf')
        
    current_threat = estimate_incoming_threat(unit, state)
    potential_threat = estimate_incoming_threat(switch_in, state)
    
    # A switch is valuable if the incoming Pokemon has higher survival probability
    # compare penalty multipliers: higher is better
    score = (potential_threat["penalty_multiplier"] - current_threat["penalty_multiplier"]) * 500.0
    
    # Type advantage shift evaluation
    # Switch is favored if switch_in resists incoming max damage types
    return max(score, 0.0)