from vgc2.battle_engine.view import StateView, BattlingPokemonView
from vgc2.battle_engine.move import BattlingMove
from vgc2.battle_engine.damage_calculator import calculate_damage
from vgc2.battle_engine.constants import BattleRuleParam
from vgc2.battle_engine.modifiers import Stat
from src.agent.battle_policy.heuristics.threat import estimate_incoming_threat

MAX_SCORE = 1000.0

def _score_single_offensive_move(attacker: BattlingPokemonView, target: BattlingPokemonView, move: BattlingMove, state: StateView, params: BattleRuleParam, attacker_side: int) -> float:
    if move.pp <= 0 or target.hp <= 0:
        return -float('inf')

    damage = calculate_damage(params=params, attacking_side=attacker_side, move=move.constants, state=state, attacker=attacker, defender=target)
    
    score = (damage / target.hp) * 100.0 if target.hp > 0 else 0.0
    
    if damage >= target.hp:
        score += 500.0
        
    return min(score, MAX_SCORE)

def _score_protect_move(unit: BattlingPokemonView, unit_side: int, state: StateView) -> float:
    threat = estimate_incoming_threat(unit, unit_side, state)
    
    if threat["is_lethal"] and threat["is_outsped"]:
        return 600.0
    
    return threat["aggro_score"] * 300.0

def _score_single_switch_action(unit: BattlingPokemonView, switch_in: BattlingPokemonView, unit_side: int, state: StateView) -> float:
    if switch_in.hp <= 0:
        return -float('inf')
        
    current_threat = estimate_incoming_threat(unit, unit_side, state)
    potential_threat = estimate_incoming_threat(switch_in, unit_side, state)
    
    score = (potential_threat["penalty_multiplier"] - current_threat["penalty_multiplier"]) * 500.0
    
    return max(score, 0.0)