from src.agent.base import Pokemon, Move
from src.agent.selection_policy.hydrator import hydrate_team


def test_hydrate_single_species_returns_populated_pokemon() -> None:
    team = hydrate_team(["snorlax"])
    assert len(team) == 1
    mon = team[0]
    assert isinstance(mon, Pokemon)
    assert mon.species == "snorlax"


def test_hydrate_single_species_has_ability() -> None:
    team = hydrate_team(["snorlax"])
    assert team[0].ability != ""


def test_hydrate_single_species_has_item() -> None:
    team = hydrate_team(["snorlax"])
    assert team[0].item != ""


def test_hydrate_single_species_has_nature() -> None:
    team = hydrate_team(["snorlax"])
    assert team[0].nature != "serious"


def test_hydrate_single_species_has_evs() -> None:
    for _ in range(5):
        team = hydrate_team(["snorlax"])
        mon = team[0]
        if mon.ev_hp + mon.ev_atk + mon.ev_def + mon.ev_spa + mon.ev_spd + mon.ev_spe > 0:
            return
    assert False, "All 5 attempts produced 0-EV Snorlax spread"


def test_hydrate_single_species_has_moves() -> None:
    team = hydrate_team(["snorlax"])
    assert len(team[0].moves) == 4
    for move in team[0].moves:
        assert isinstance(move, Move)
        assert move.name != ""


def test_hydrate_six_species_returns_six() -> None:
    species = ["snorlax", "blastoise", "venusaur", "gengar", "conkeldurr", "clefable"]
    team = hydrate_team(species)
    assert len(team) == 6
    for mon in team:
        assert isinstance(mon, Pokemon)
        assert mon.species != ""


def test_hydrate_all_pokemon_have_items_and_abilities() -> None:
    species = ["snorlax", "blastoise", "venusaur", "gengar", "conkeldurr", "clefable"]
    team = hydrate_team(species)
    for mon in team:
        assert mon.ability != "", f"{mon.species} missing ability"
        assert mon.item != "", f"{mon.species} missing item"
        assert len(mon.moves) == 4, f"{mon.species} has {len(mon.moves)} moves"


def test_hydrate_with_variance_applied() -> None:
    results: set[str] = set()
    for _ in range(20):
        team = hydrate_team(["snorlax"])
        results.add(team[0].item)
    assert len(results) > 1, f"only got one item: {results}"


def test_hydrate_unknown_species_fills_placeholder() -> None:
    team = hydrate_team(["not_a_real_pokemon_xyz"])
    assert len(team) == 1
    assert team[0].species == "not_a_real_pokemon_xyz"


def test_hydrate_empty_list_returns_empty() -> None:
    team = hydrate_team([])
    assert team == []


def test_hydrate_produces_poke_env_compatible() -> None:
    species = ["snorlax", "blastoise", "venusaur"]
    team = hydrate_team(species)
    for mon in team:
        assert hasattr(mon, "species")
        assert hasattr(mon, "ability")
        assert hasattr(mon, "item")
        assert hasattr(mon, "nature")
        assert hasattr(mon, "level")
        assert hasattr(mon, "ev_hp")
        assert hasattr(mon, "ev_atk")
        assert hasattr(mon, "ev_def")
        assert hasattr(mon, "ev_spa")
        assert hasattr(mon, "ev_spd")
        assert hasattr(mon, "ev_spe")
        assert hasattr(mon, "moves")
        assert all(hasattr(m, "name") for m in mon.moves)
