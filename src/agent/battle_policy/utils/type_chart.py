from typing import Optional
from vgc2.battle_engine.modifiers import Type

_EFFECTIVENESS_MATRIX = {
    Type.NORMAL: {Type.ROCK: 0.5, Type.GHOST: 0.0, Type.STEEL: 0.5},
    Type.FIRE: {Type.FIRE: 0.5, Type.WATER: 0.5, Type.GRASS: 2.0, Type.ICE: 2.0, Type.BUG: 2.0, Type.ROCK: 0.5, Type.DRAGON: 0.5, Type.STEEL: 2.0},
    Type.WATER: {Type.FIRE: 2.0, Type.WATER: 0.5, Type.GRASS: 0.5, Type.GROUND: 2.0, Type.ROCK: 2.0, Type.DRAGON: 0.5},
    Type.ELECTRIC: {Type.WATER: 2.0, Type.ELECTRIC: 0.5, Type.GRASS: 0.5, Type.GROUND: 0.0, Type.FLYING: 2.0, Type.DRAGON: 0.5},
    Type.GRASS: {Type.FIRE: 0.5, Type.WATER: 2.0, Type.GRASS: 0.5, Type.POISON: 0.5, Type.GROUND: 2.0, Type.FLYING: 0.5, Type.BUG: 0.5, Type.ROCK: 2.0, Type.DRAGON: 0.5, Type.STEEL: 0.5},
    Type.ICE: {Type.FIRE: 0.5, Type.WATER: 0.5, Type.GRASS: 2.0, Type.ICE: 0.5, Type.GROUND: 2.0, Type.FLYING: 2.0, Type.DRAGON: 2.0, Type.STEEL: 0.5},
    Type.FIGHT: {Type.NORMAL: 2.0, Type.ICE: 2.0, Type.POISON: 0.5, Type.FLYING: 0.5, Type.PSYCHIC: 0.5, Type.BUG: 0.5, Type.ROCK: 2.0, Type.GHOST: 0.0, Type.DARK: 2.0, Type.STEEL: 2.0, Type.FAIRY: 0.5},
    Type.POISON: {Type.GRASS: 2.0, Type.POISON: 0.5, Type.GROUND: 0.5, Type.ROCK: 0.5, Type.GHOST: 0.5, Type.STEEL: 0.0, Type.FAIRY: 2.0},
    Type.GROUND: {Type.FIRE: 2.0, Type.ELECTRIC: 2.0, Type.POISON: 2.0, Type.FLYING: 0.0, Type.BUG: 0.5, Type.ROCK: 2.0, Type.STEEL: 2.0, Type.GRASS: 0.5},
    Type.FLYING: {Type.ELECTRIC: 0.5, Type.GRASS: 2.0, Type.FIGHT: 2.0, Type.BUG: 2.0, Type.ROCK: 0.5, Type.STEEL: 0.5},
    Type.PSYCHIC: {Type.FIGHT: 2.0, Type.POISON: 2.0, Type.PSYCHIC: 0.5, Type.DARK: 0.0, Type.STEEL: 0.5},
    Type.BUG: {Type.FIRE: 0.5, Type.GRASS: 2.0, Type.FIGHT: 0.5, Type.POISON: 0.5, Type.FLYING: 0.5, Type.PSYCHIC: 2.0, Type.GHOST: 0.5, Type.DARK: 2.0, Type.STEEL: 0.5, Type.FAIRY: 0.5},
    Type.ROCK: {Type.FIRE: 2.0, Type.ICE: 2.0, Type.FIGHT: 0.5, Type.GROUND: 0.5, Type.FLYING: 2.0, Type.BUG: 2.0, Type.STEEL: 0.5},
    Type.GHOST: {Type.NORMAL: 0.0, Type.PSYCHIC: 2.0, Type.GHOST: 2.0, Type.DARK: 0.5},
    Type.DRAGON: {Type.DRAGON: 2.0, Type.STEEL: 0.5, Type.FAIRY: 0.0},
    Type.DARK: {Type.FIGHT: 0.5, Type.PSYCHIC: 2.0, Type.GHOST: 2.0, Type.DARK: 0.5, Type.FAIRY: 0.5},
    Type.STEEL: {Type.FIRE: 0.5, Type.WATER: 0.5, Type.ELECTRIC: 0.5, Type.ICE: 2.0, Type.ROCK: 2.0, Type.STEEL: 0.5, Type.FAIRY: 2.0},
    Type.FAIRY: {Type.FIRE: 0.5, Type.FIGHT: 2.0, Type.POISON: 0.5, Type.DRAGON: 2.0, Type.DARK: 2.0, Type.STEEL: 0.5}
}

def get_type_multiplier(attack_type: Type, defense_type_1: Type, defense_type_2: Optional[Type] = None) -> float:
    modifier_1 = _EFFECTIVENESS_MATRIX.get(attack_type, {}).get(defense_type_1, 1.0)
    modifier_2 = _EFFECTIVENESS_MATRIX.get(attack_type, {}).get(defense_type_2, 1.0) if defense_type_2 else 1.0
    return modifier_1 * modifier_2