from __future__ import annotations
import pytest
from unittest.mock import Mock, patch
from src.agent.selection_policy.heuristics.scoring import (
    score_move,
    calculate_utility_score,
    calculate_damage_score,
    _calculate_protect_score,
    _calculate_status_score,
    _calculate_field_effect_score,
)
from src.config.selection_model import SelectionHeuristicsConfig
from vgc2.battle_engine.modifiers import Stat, Type, Category, Status, Weather, Terrain

# --- Fixtures ---

@pytest.fixture
def cfg() -> SelectionHeuristicsConfig:
    return SelectionHeuristicsConfig()

@pytest.fixture
def params() -> Mock:
    return Mock()

@pytest.fixture
def species() -> Mock:
    s = Mock()
    s.base_stats = {
        Stat.MAX_HP: 300, Stat.ATTACK: 120, Stat.SPECIAL_ATTACK: 100,
        Stat.SPEED: 110, Stat.DEFENSE: 100, Stat.SPECIAL_DEFENSE: 100
    }
    s.types = []
    s.moves = []
    return s

@pytest.fixture
def my_team(species: Mock) -> Mock:
    t = Mock()
    p1 = Mock()
    p1.stats = {Stat.MAX_HP: 300, Stat.ATTACK: 130, Stat.SPECIAL_ATTACK: 90, Stat.SPEED: 100}
    p1.species = species
    p1.moves = [Mock(base_power=80, category=Category.PHYSICAL, pkm_type=Type.NORMAL)]
    
    p2 = Mock()
    p2.stats = {Stat.MAX_HP: 300, Stat.ATTACK: 90, Stat.SPECIAL_ATTACK: 130, Stat.SPEED: 140}
    p2.species = species
    p2.moves = [Mock(base_power=90, category=Category.SPECIAL, pkm_type=Type.NORMAL)]
    
    t.members = [p1, p2]
    return t

@pytest.fixture
def opp_views(species: Mock) -> list[Mock]:
    v = Mock()
    v.species = species
    v.moves = [Mock(base_power=80, pkm_type=Type.WATER, category=Category.PHYSICAL)]
    return [v, v]

# --- Unit Tests ---

class TestProtectScore:
    def test_returns_positive_float_on_damage_threat(self, species, my_team, params):
        move = Mock(base_power=0, category=Category.OTHER, protect=True)
        with patch("src.agent.selection_policy.heuristics.scoring.calculate_damage", return_value=150.0):
            result = _calculate_protect_score(move, species, my_team, params)
            assert result == pytest.approx(50.0)

    def test_zero_on_empty_hp_species(self, my_team, params):
        move = Mock(base_power=0, category=Category.OTHER, protect=True)
        species = Mock()
        species.base_stats = {Stat.MAX_HP: 0}
        species.moves = []
        with patch("src.agent.selection_policy.heuristics.scoring.calculate_damage", return_value=100.0):
            assert _calculate_protect_score(move, species, my_team, params) == 0.0

class TestStatusScore:
    def test_toxic_uses_config_coefficient(self, species, my_team, opp_views, params, cfg):
        move = Mock(base_power=0, category=Category.OTHER, status=Status.TOXIC)
        result = _calculate_status_score(move, species, my_team, opp_views, params, cfg)
        assert result == pytest.approx(cfg.toxic_damage_coefficient * 100)

    def test_burn_scales_with_mitigation(self, species, my_team, opp_views, params, cfg):
        move = Mock(base_power=0, category=Category.OTHER, status=Status.BURN)
        with patch("src.agent.selection_policy.heuristics.scoring.calculate_damage", side_effect=[100.0, 50.0]):
            result = _calculate_status_score(move, species, my_team, opp_views, params, cfg)
            assert result > 0.0

    def test_paralysis_denial_calculation(self, species, my_team, opp_views, params, cfg):
        move = Mock(base_power=0, category=Category.OTHER, status=Status.PARALYZED)
        with patch("src.agent.selection_policy.heuristics.scoring.calculate_damage", return_value=80.0):
            result = _calculate_status_score(move, species, my_team, opp_views, params, cfg)
            assert result == pytest.approx((80.0 * cfg.paralysis_denial_chance / 300) * 100)

    def test_sleep_denial_turns(self, species, my_team, opp_views, params, cfg):
        species.moves = [Mock(base_power=100)]
        move = Mock(base_power=0, category=Category.OTHER, status=Status.SLEEP)
        with patch("src.agent.selection_policy.heuristics.scoring.calculate_damage", return_value=90.0):
            result = _calculate_status_score(move, species, my_team, opp_views, params, cfg)
            assert result == pytest.approx((90.0 * cfg.sleep_denial_turns / 300) * 100)

class TestFieldEffectScore:
    def test_rain_weather_swing(self, my_team, opp_views, params, cfg):
        move = Mock(weather_start=Weather.RAIN, field_start=None)
        with patch("src.agent.selection_policy.heuristics.scoring.calculate_damage", return_value=50.0):
            result = _calculate_field_effect_score(move, my_team, opp_views, params, cfg)
            assert isinstance(result, float)

    def test_terrain_terrain_swing(self, my_team, opp_views, params, cfg):
        move = Mock(weather_start=None, field_start=Terrain.ELECTRIC_TERRAIN)
        with patch("src.agent.selection_policy.heuristics.scoring.calculate_damage", return_value=40.0):
            result = _calculate_field_effect_score(move, my_team, opp_views, params, cfg)
            assert isinstance(result, float)

    def test_snow_passive_damage(self, my_team, opp_views, params, cfg):
        move = Mock(weather_start=Weather.SNOW, field_start=None)
        result = _calculate_field_effect_score(move, my_team, opp_views, params, cfg)
        assert result >= 0.0

class TestCalculateUtilityScore:
    def test_routes_to_protect(self, move: Mock, species, my_team, opp_views, params, cfg):
        move = Mock(base_power=0, category=Category.OTHER, protect=True, status=None, weather_start=None, field_start=None)
        with patch("src.agent.selection_policy.heuristics.scoring._calculate_protect_score", return_value=25.0) as mock_fn:
            assert calculate_utility_score(move, species, my_team, opp_views, params, cfg) == 25.0
            mock_fn.assert_called_once()

    def test_routes_to_status(self, species, my_team, opp_views, params, cfg):
        move = Mock(base_power=0, category=Category.OTHER, protect=False, status=Status.BURN, weather_start=None, field_start=None)
        with patch("src.agent.selection_policy.heuristics.scoring._calculate_status_score", return_value=15.0) as mock_fn:
            assert calculate_utility_score(move, species, my_team, opp_views, params, cfg) == 15.0

    def test_routes_to_field(self, species, my_team, opp_views, params, cfg):
        move = Mock(base_power=0, category=Category.OTHER, protect=False, status=None, weather_start=Weather.RAIN, field_start=None)
        with patch("src.agent.selection_policy.heuristics.scoring._calculate_field_effect_score", return_value=10.0) as mock_fn:
            assert calculate_utility_score(move, species, my_team, opp_views, params, cfg) == 10.0

    def test_returns_zero_for_damaging_moves(self, species, my_team, opp_views, params, cfg):
        move = Mock(base_power=90, category=Category.PHYSICAL)
        assert calculate_utility_score(move, species, my_team, opp_views, params, cfg) == 0.0

class TestCalculateDamageScore:
    def test_zero_for_non_damaging(self, species, my_team, params, cfg):
        move = Mock(base_power=0, category=Category.OTHER)
        assert calculate_damage_score(move, species, my_team, params, cfg) == 0.0

    def test_returns_float_for_damaging(self, species, my_team, params, cfg):
        move = Mock(base_power=90, category=Category.PHYSICAL)
        species.moves = [move]
        with patch("src.agent.selection_policy.heuristics.scoring.create_archetype_builds", return_value=[Mock()]):
            with patch("src.agent.selection_policy.heuristics.scoring.calculate_damage", return_value=40.0):
                result = calculate_damage_score(move, species, my_team, params, cfg)
                assert isinstance(result, float)
                assert result >= 0.0

class TestScoreMoveRouter:
    def test_utility_dispatch(self, species, my_team, opp_views, params, cfg):
        move = Mock(base_power=0, category=Category.OTHER, protect=True)
        with patch("src.agent.selection_policy.heuristics.scoring.calculate_utility_score", return_value=20.0) as m:
            score_move(move, species, my_team, opp_views, params, cfg)
            m.assert_called_once()

    def test_damage_dispatch(self, species, my_team, opp_views, params, cfg):
        move = Mock(base_power=100, category=Category.SPECIAL)
        with patch("src.agent.selection_policy.heuristics.scoring.calculate_damage_score", return_value=75.0) as m:
            score_move(move, species, my_team, opp_views, params, cfg)
            m.assert_called_once()

class TestPropertyConstraints:
    @pytest.mark.parametrize("base_power,category", [(0, Category.OTHER), (90, Category.PHYSICAL), (0, Category.PHYSICAL)])
    def test_no_nan_inf_returns(self, base_power, category, species, my_team, opp_views, params, cfg):
        move = Mock(base_power=base_power, category=category, protect=False, status=None, weather_start=None, field_start=None)
        species.moves = [move] if base_power > 0 else []
        
        with patch("src.agent.selection_policy.heuristics.scoring.calculate_damage", return_value=0.0):
            with patch("src.agent.selection_policy.heuristics.scoring.create_archetype_builds", return_value=[Mock()] if base_power > 0 else []):
                result = score_move(move, species, my_team, opp_views, params, cfg)
                assert not (result != result)
                assert result != float("inf")
                assert result != float("-inf")