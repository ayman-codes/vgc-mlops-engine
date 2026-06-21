from dataclasses import dataclass, field


@dataclass
class Move:
    name: str
    base_power: int = 0
    move_type: str = "normal"
    damage_class: str = "physical"
    accuracy: float = 1.0
    priority: int = 0
    current_pp: int = 5


@dataclass
class Pokemon:
    species: str
    pokeapi_id: int = 0
    level: int = 50
    hp: int = 100
    max_hp: int = 100
    attack: int = 100
    defense: int = 100
    special_attack: int = 100
    special_defense: int = 100
    speed: int = 100
    type_1: str = "normal"
    type_2: str | None = None
    ability: str = ""
    item: str = ""
    tera_type: str = ""
    status: str = ""
    moves: list[Move] = field(default_factory=list)
    ev_hp: int = 0
    ev_atk: int = 0
    ev_def: int = 0
    ev_spa: int = 0
    ev_spd: int = 0
    ev_spe: int = 0
    iv_hp: int = 31
    iv_atk: int = 31
    iv_def: int = 31
    iv_spa: int = 31
    iv_spd: int = 31
    iv_spe: int = 31
    nature: str = "serious"


@dataclass
class Team:
    pokemon: list[Pokemon] = field(default_factory=list)
    active_index: int = 0

    @property
    def active(self) -> Pokemon | None:
        if 0 <= self.active_index < len(self.pokemon):
            return self.pokemon[self.active_index]
        return None
