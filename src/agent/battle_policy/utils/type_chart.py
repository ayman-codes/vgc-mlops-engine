from typing import Optional
from vgc2.battle_engine.constants import PokemonType

_EFFECTIVENESS_MATRIX = {
    PokemonType.NORMAL: {PokemonType.ROCK: 0.5, PokemonType.GHOST: 0.0, PokemonType.STEEL: 0.5},
    PokemonType.FIRE: {PokemonType.FIRE: 0.5, PokemonType.WATER: 0.5, PokemonType.GRASS: 2.0, PokemonType.ICE: 2.0, PokemonType.BUG: 2.0, PokemonType.ROCK: 0.5, PokemonType.DRAGON: 0.5, PokemonType.STEEL: 2.0},
    PokemonType.WATER: {PokemonType.FIRE: 2.0, PokemonType.WATER: 0.5, PokemonType.GRASS: 0.5, PokemonType.GROUND: 2.0, PokemonType.ROCK: 2.0, PokemonType.DRAGON: 0.5},
    PokemonType.ELECTRIC: {PokemonType.WATER: 2.0, PokemonType.ELECTRIC: 0.5, PokemonType.GRASS: 0.5, PokemonType.GROUND: 0.0, PokemonType.FLYING: 2.0, PokemonType.DRAGON: 0.5},
    PokemonType.GRASS: {PokemonType.FIRE: 0.5, PokemonType.WATER: 2.0, PokemonType.GRASS: 0.5, PokemonType.POISON: 0.5, PokemonType.GROUND: 2.0, PokemonType.FLYING: 0.5, PokemonType.BUG: 0.5, PokemonType.ROCK: 2.0, PokemonType.DRAGON: 0.5, PokemonType.STEEL: 0.5},
    PokemonType.ICE: {PokemonType.FIRE: 0.5, PokemonType.WATER: 0.5, PokemonType.GRASS: 2.0, PokemonType.ICE: 0.5, PokemonType.GROUND: 2.0, PokemonType.FLYING: 2.0, PokemonType.DRAGON: 2.0, PokemonType.STEEL: 0.5},
    PokemonType.FIGHTING: {PokemonType.NORMAL: 2.0, PokemonType.ICE: 2.0, PokemonType.POISON: 0.5, PokemonType.FLYING: 0.5, PokemonType.PSYCHIC: 0.5, PokemonType.BUG: 0.5, PokemonType.ROCK: 2.0, PokemonType.GHOST: 0.0, PokemonType.DARK: 2.0, PokemonType.STEEL: 2.0, PokemonType.FAIRY: 0.5},
    PokemonType.POISON: {PokemonType.GRASS: 2.0, PokemonType.POISON: 0.5, PokemonType.GROUND: 0.5, PokemonType.ROCK: 0.5, PokemonType.GHOST: 0.5, PokemonType.STEEL: 0.0, PokemonType.FAIRY: 2.0},
    PokemonType.GROUND: {PokemonType.FIRE: 2.0, PokemonType.ELECTRIC: 2.0, PokemonType.POISON: 2.0, PokemonType.FLYING: 0.0, PokemonType.BUG: 0.5, PokemonType.ROCK: 2.0, PokemonType.STEEL: 2.0, PokemonType.GRASS: 0.5},
    PokemonType.FLYING: {PokemonType.ELECTRIC: 0.5, PokemonType.GRASS: 2.0, PokemonType.FIGHTING: 2.0, PokemonType.BUG: 2.0, PokemonType.ROCK: 0.5, PokemonType.STEEL: 0.5},
    PokemonType.PSYCHIC: {PokemonType.FIGHTING: 2.0, PokemonType.POISON: 2.0, PokemonType.PSYCHIC: 0.5, PokemonType.DARK: 0.0, PokemonType.STEEL: 0.5},
    PokemonType.BUG: {PokemonType.FIRE: 0.5, PokemonType.GRASS: 2.0, PokemonType.FIGHTING: 0.5, PokemonType.POISON: 0.5, PokemonType.FLYING: 0.5, PokemonType.PSYCHIC: 2.0, PokemonType.GHOST: 0.5, PokemonType.DARK: 2.0, PokemonType.STEEL: 0.5, PokemonType.FAIRY: 0.5},
    PokemonType.ROCK: {PokemonType.FIRE: 2.0, PokemonType.ICE: 2.0, PokemonType.FIGHTING: 0.5, PokemonType.GROUND: 0.5, PokemonType.FLYING: 2.0, PokemonType.BUG: 2.0, PokemonType.STEEL: 0.5},
    PokemonType.GHOST: {PokemonType.NORMAL: 0.0, PokemonType.PSYCHIC: 2.0, PokemonType.GHOST: 2.0, PokemonType.DARK: 0.5},
    PokemonType.DRAGON: {PokemonType.DRAGON: 2.0, PokemonType.STEEL: 0.5, PokemonType.FAIRY: 0.0},
    PokemonType.DARK: {PokemonType.FIGHTING: 0.5, PokemonType.PSYCHIC: 2.0, PokemonType.GHOST: 2.0, PokemonType.DARK: 0.5, PokemonType.FAIRY: 0.5},
    PokemonType.STEEL: {PokemonType.FIRE: 0.5, PokemonType.WATER: 0.5, PokemonType.ELECTRIC: 0.5, PokemonType.ICE: 2.0, PokemonType.ROCK: 2.0, PokemonType.STEEL: 0.5, PokemonType.FAIRY: 2.0},
    PokemonType.FAIRY: {PokemonType.FIRE: 0.5, PokemonType.FIGHTING: 2.0, PokemonType.POISON: 0.5, PokemonType.DRAGON: 2.0, PokemonType.DARK: 2.0, PokemonType.STEEL: 0.5}
}

def get_type_multiplier(attack_type: PokemonType, defense_type_1: PokemonType, defense_type_2: Optional[PokemonType] = None) -> float:
    modifier_1 = _EFFECTIVENESS_MATRIX.get(attack_type, {}).get(defense_type_1, 1.0)
    modifier_2 = _EFFECTIVENESS_MATRIX.get(attack_type, {}).get(defense_type_2, 1.0) if defense_type_2 else 1.0
    return modifier_1 * modifier_2