"""Type chart and damage calculation utilities for Generation 9.

Provides the 18-type effectiveness matrix and the standard damage formula
used by the selection pipeline for strategy screening.
"""

TYPE_CHART: dict[str, dict[str, float]] = {
    "normal": {"rock": 0.5, "ghost": 0.0, "steel": 0.5},
    "fire": {"fire": 0.5, "water": 0.5, "grass": 2.0, "ice": 2.0, "bug": 2.0, "rock": 0.5, "dragon": 0.5, "steel": 2.0},
    "water": {"fire": 2.0, "water": 0.5, "grass": 0.5, "ground": 2.0, "rock": 2.0, "dragon": 0.5},
    "electric": {"water": 2.0, "electric": 0.5, "grass": 0.5, "ground": 0.0, "flying": 2.0, "dragon": 0.5},
    "grass": {"fire": 0.5, "water": 2.0, "grass": 0.5, "poison": 0.5, "ground": 2.0, "flying": 0.5, "bug": 0.5, "rock": 2.0, "dragon": 0.5, "steel": 0.5},
    "ice": {"fire": 0.5, "water": 0.5, "grass": 2.0, "ice": 0.5, "ground": 2.0, "flying": 2.0, "dragon": 2.0, "steel": 0.5},
    "fighting": {"normal": 2.0, "ice": 2.0, "poison": 0.5, "flying": 0.5, "psychic": 0.5, "bug": 0.5, "rock": 2.0, "ghost": 0.0, "dark": 2.0, "steel": 2.0, "fairy": 0.5},
    "poison": {"grass": 2.0, "poison": 0.5, "ground": 0.5, "rock": 0.5, "ghost": 0.5, "steel": 0.0, "fairy": 2.0},
    "ground": {"fire": 2.0, "grass": 0.5, "electric": 2.0, "poison": 2.0, "flying": 0.0, "bug": 0.5, "rock": 2.0, "steel": 2.0},
    "flying": {"electric": 0.5, "grass": 2.0, "fighting": 2.0, "bug": 2.0, "rock": 0.5, "steel": 0.5},
    "psychic": {"fighting": 2.0, "poison": 2.0, "psychic": 0.5, "dark": 0.0, "steel": 0.5},
    "bug": {"fire": 0.5, "grass": 2.0, "fighting": 0.5, "poison": 0.5, "flying": 0.5, "psychic": 2.0, "ghost": 0.5, "dark": 2.0, "steel": 0.5, "fairy": 0.5},
    "rock": {"fire": 2.0, "ice": 2.0, "fighting": 0.5, "ground": 0.5, "flying": 2.0, "bug": 2.0, "steel": 0.5},
    "ghost": {"normal": 0.0, "psychic": 2.0, "ghost": 2.0, "dark": 0.5},
    "dragon": {"fire": 0.5, "water": 0.5, "electric": 0.5, "grass": 0.5, "dragon": 2.0, "steel": 0.5, "fairy": 0.0},
    "dark": {"fighting": 0.5, "psychic": 2.0, "ghost": 2.0, "dark": 0.5, "fairy": 0.5},
    "steel": {"fire": 0.5, "water": 0.5, "electric": 0.5, "ice": 2.0, "rock": 2.0, "steel": 0.5, "fairy": 2.0},
    "fairy": {"fire": 0.5, "poison": 0.5, "fighting": 2.0, "dragon": 2.0, "dark": 2.0, "steel": 0.5},
}


def type_effectiveness(move_type: str, defender_type: str) -> float:
    """Look up the type effectiveness multiplier between a move and a defender type.

    Args:
        move_type: The attacking move's type (lowercase, e.g. "fire").
        defender_type: The defender's type (lowercase, e.g. "grass").

    Returns:
        Effectiveness multiplier (0.0, 0.5, 1.0, or 2.0).
    """
    return TYPE_CHART.get(move_type, {}).get(defender_type, 1.0)


def calculate_damage(
    level: int,
    power: int,
    attack: int,
    defense: int,
    modifier: float = 1.0,
) -> float:
    """Base Generation 9 damage formula without type interaction.

    Args:
        level: Attacker's level (1–100).
        power: Move base power.
        attack: Attacker's offensive stat (Atk or SpA).
        defense: Defender's defensive stat (Def or SpD).
        modifier: Final accumulated multiplier (STAB, type effectiveness, etc.).

    Returns:
        Rounded damage expectation as a float.
    """
    return ((2 * level / 5 + 2) * power * attack / defense / 50 + 2) * modifier


def calculate_damage_with_types(
    level: int,
    power: int,
    attack: int,
    defense: int,
    move_type: str,
    defender_types: list[str],
    stab: float = 1.0,
) -> float:
    """Damage calculation combining base formula, type chart, and STAB.

    Args:
        level: Attacker's level.
        power: Move base power.
        attack: Attacker's offensive stat.
        defense: Defender's defensive stat.
        move_type: Move type for type-chart lookup.
        defender_types: List of the defender's types (1 or 2 elements).
        stab: Same-type attack bonus (default 1.0, typically 1.5).

    Returns:
        Final damage after type effectiveness and STAB.
    """
    type_eff = 1.0
    for t in defender_types:
        type_eff *= type_effectiveness(move_type, t)
    modifier = stab * type_eff
    return calculate_damage(level, power, attack, defense, modifier)
