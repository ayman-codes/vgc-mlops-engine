import pytest
from unittest.mock import MagicMock, patch
from vgc2.battle_engine.modifiers import Status, Weather, Category
from src.config.selection_model import SelectionConfig
from src.agent.selection_policy.heuristics.scoring import calculate_utility_score

def create_mock_move(base_power: int = 0, category: Category = Category.OTHER, status: Status = None, protect: bool = False, weather: Weather = None) -> MagicMock:
    move = MagicMock()
    move.base_power = base_power
    move.category = category
    move.status = status
    move.protect = protect
    move.weather_start = weather
    move.field_start = None
    move.priority = 0
    return move

def test_calculate_utility_score_protect() -> None:
    config = SelectionConfig()
    move = create_mock_move(protect=True)
    
    attacker_species = MagicMock()
    # Provide all 6 stats: (HP, Atk, Def, SpA, SpD, Spe)
    attacker_species.base_stats = (100, 100, 100, 100, 100, 100)
    
    my_pkm = MagicMock()
    my_pkm.moves = [create_mock_move(base_power=50, category=Category.PHYSICAL)]
    my_full_team = MagicMock()
    my_full_team.members = [my_pkm]
    
    battle_params = MagicMock()
    all_opp_species_views = []
    
    with patch("src.agent.selection_policy.heuristics.scoring.calculate_damage", return_value=25.0):
        score = calculate_utility_score(move, attacker_species, my_full_team, all_opp_species_views, battle_params, config)
    
    assert score == 25.0

def test_calculate_utility_score_toxic() -> None:
    config = SelectionConfig()
    move = create_mock_move(status=Status.TOXIC)
    
    attacker_species = MagicMock()
    attacker_species.base_stats = (100, 100, 100, 100, 100, 100)
    
    my_full_team = MagicMock()
    my_full_team.members = []
    
    score = calculate_utility_score(move, attacker_species, my_full_team, [], MagicMock(), config)
    assert score == pytest.approx(62.5)

if __name__ == "__main__":
    pytest.main([__file__])