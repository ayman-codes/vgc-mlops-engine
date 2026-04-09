import pytest
from unittest.mock import MagicMock, patch
from vgc2.battle_engine.modifiers import Nature, Category
from src.config.selection_model import SelectionConfig
from src.agent.selection_policy.heuristics.archetype import create_archetype_builds, predict_moveset

def create_mock_move(name: str, category: Category, base_power: int = 50) -> MagicMock:
    move = MagicMock()
    move.name = name
    move.category = category
    move.base_power = base_power
    return move

def test_create_archetype_builds_physical() -> None:
    config = SelectionConfig()
    species = MagicMock()
    # (HP, Atk, Def, SpA, SpD, Spe) - Physical leaning
    species.base_stats = (100, 130, 80, 60, 80, 100)
    
    move1 = create_mock_move("Tackle", Category.PHYSICAL)
    species.moves = [move1]
    
    with patch("src.agent.selection_policy.heuristics.archetype.Pokemon") as mock_pokemon:
        builds = create_archetype_builds(species, [move1], config)
        
    assert len(builds) == 3
    mock_pokemon.assert_any_call(species=species, move_indexes=[0], nature=Nature.JOLLY, evs=(4, 252, 0, 0, 0, 252))

def test_create_archetype_builds_mixed() -> None:
    config = SelectionConfig()
    species = MagicMock()
    # (HP, Atk, Def, SpA, SpD, Spe) - Mixed stats
    species.base_stats = (100, 100, 80, 95, 80, 100)
    
    move1 = create_mock_move("Tackle", Category.PHYSICAL)
    species.moves = [move1]
    
    with patch("src.agent.selection_policy.heuristics.archetype.Pokemon"):
        builds = create_archetype_builds(species, [move1], config)
        
    assert len(builds) == 4

@patch("src.agent.selection_policy.heuristics.archetype.calculate_damage", return_value=50.0)
@patch("src.agent.selection_policy.heuristics.archetype.calculate_utility_score", return_value=10.0)
def test_predict_moveset_ranking(mock_utility, mock_damage) -> None:
    config = SelectionConfig()
    species = MagicMock()
    species.base_stats = (100, 100, 100, 100, 100, 100)
    
    moves = [
        create_mock_move("M1", Category.PHYSICAL),
        create_mock_move("M2", Category.SPECIAL),
        create_mock_move("M3", Category.OTHER, base_power=0),
        create_mock_move("M4", Category.PHYSICAL),
        create_mock_move("M5", Category.SPECIAL)
    ]
    species.moves = moves
    
    my_pkm = MagicMock()
    my_pkm.stats = (100, 100, 100, 100, 100, 100)
    my_team = MagicMock()
    my_team.members = [my_pkm]
    
    with patch("src.agent.selection_policy.heuristics.archetype.create_archetype_builds", return_value=[MagicMock()]):
        top_moves = predict_moveset(species, my_team, [], MagicMock(), config)
        
    assert len(top_moves) == 4

if __name__ == "__main__":
    pytest.main([__file__])