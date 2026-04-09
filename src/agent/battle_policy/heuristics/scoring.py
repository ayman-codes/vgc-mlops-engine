from vgc2.battle_engine.view import StateView, BattlingPokemonView
from vgc2.battle_engine.move import BattlingMove
from vgc2.battle_engine.damage_calculator import calculate_damage
from vgc2.battle_engine.constants import BattleRuleParam
from vgc2.battle_engine.modifiers import Category, Status, Stat, Terrain
from src.agent.battle_policy.heuristics.threat import estimate_incoming_threat

def _score_single_offensive_move(attacker: BattlingPokemonView, target: BattlingPokemonView, move: BattlingMove, state: StateView, params: BattleRuleParam, attacker_side: int) -> float:
    if move.pp <= 0 or target.hp <= 0:
        return -float('inf')

    target_threat = estimate_incoming_threat(attacker, attacker_side, state, params)
    score = 0.0

    if move.constants.category != Category.OTHER and move.constants.base_power > 0:
        damage = calculate_damage(params=params, attacking_side=attacker_side, move=move.constants, state=state, attacker=attacker, defender=target)
        actual_damage_dealt = min(damage, float(target.hp))
        score += actual_damage_dealt
        
        if damage >= target.hp:
            score += target_threat["max_incoming_damage"]
            
    else:
        if move.constants.status == Status.SLEEP:
            score += target_threat["max_incoming_damage"] * 1.5 
        elif move.constants.status == Status.PARALYZED:
            score += target_threat["max_incoming_damage"] * params.PARALYSIS_THRESHOLD
        elif move.constants.status == Status.BURN:
            score += target_threat["max_incoming_damage"] * params.BURN_DAMAGE_MODIFIER
        
        if move.constants.toggle_tailwind and target_threat["is_outsped"]:
            score += target_threat["max_incoming_damage"]
            
        if move.constants.toggle_reflect or move.constants.toggle_lightscreen:
            score += target_threat["max_incoming_damage"] * 0.5
            
        if move.constants.heal > 0:
            max_hp = attacker.constants.stats[Stat.MAX_HP]
            missing_hp = max_hp - attacker.hp
            score += min(missing_hp, max_hp * move.constants.heal)

        if move.constants.toggle_trickroom and target_threat["is_outsped"]:
            score += target_threat["max_incoming_damage"]

        if move.constants.field_start != Terrain.NONE:
            score += target_threat["max_incoming_damage"] * 0.25

        if move.constants.self_boosts and any(b > 0 for b in move.constants.boosts):
            score += target_threat["max_incoming_damage"] * 0.5
        elif not move.constants.self_boosts and any(b < 0 for b in move.constants.boosts):
            score += target_threat["max_incoming_damage"] * 0.5

    acc = move.constants.accuracy if move.constants.accuracy else 1.0
    return score * acc

def _score_protect_move(unit: BattlingPokemonView, unit_side: int, state: StateView, params: BattleRuleParam) -> float:
    threat = estimate_incoming_threat(unit, unit_side, state, params)
    return float(threat["max_incoming_damage"])

def _score_single_switch_action(unit: BattlingPokemonView, switch_in: BattlingPokemonView, unit_side: int, state: StateView, params: BattleRuleParam) -> float:
    if switch_in.hp <= 0:
        return -float('inf')
        
    current_threat = estimate_incoming_threat(unit, unit_side, state, params)
    potential_threat = estimate_incoming_threat(switch_in, unit_side, state, params)
    
    damage_mitigated = current_threat["max_incoming_damage"] - potential_threat["max_incoming_damage"]
    return float(damage_mitigated)