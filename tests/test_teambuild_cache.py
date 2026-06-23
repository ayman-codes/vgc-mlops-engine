import json
import os
import tempfile
from typing import Any
import joblib
import numpy as np
import pytest
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.agent.teambuild_policy.cache import (
    _parse_spread,
    _weighted_choice,
    _top_weighted_moves,
    TeambuildCache,
    load_teambuild_cache,
)
from src.agent.base import Pokemon
from src.config.teambuild_config import TeambuildConfig


def _build_mock_smogon_json() -> dict[str, Any]:
    """Build a minimal Smogon Chaos JSON with 5 species for testing.

    Returns:
        Dict with top-level "data" key containing 5 species entries.
    """
    return {
        "data": {
            "FlutterMane": {
                "usage": 0.55,
                "Raw count": 220000,
                "Abilities": {"protosynthesis": 100.0},
                "Items": {"boosterenergy": 60.0, "focussash": 40.0},
                "Spreads": {
                    "Timid:0/0/0/32/0/32": 80.0,
                    "Modest:0/0/0/32/0/32": 20.0,
                },
                "Moves": {
                    "shadowball": 90.0,
                    "moonblast": 85.0,
                    "protect": 70.0,
                    "thunderbolt": 45.0,
                    "energyball": 15.0,
                },
            },
            "Incineroar": {
                "usage": 0.32,
                "Raw count": 180000,
                "Abilities": {"intimidate": 95.0, "blaze": 5.0},
                "Items": {"sitrusberry": 50.0, "safetygoggles": 30.0},
                "Spreads": {"Careful:32/0/24/0/8/0": 90.0},
                "Moves": {
                    "fakeout": 95.0,
                    "flareblitz": 80.0,
                    "knockoff": 75.0,
                    "partingshot": 65.0,
                    "uturn": 20.0,
                },
            },
            "Garchomp": {
                "usage": 0.48,
                "Raw count": 220000,
                "Abilities": {"roughskin": 100.0},
                "Items": {"lifeorb": 55.0, "focussash": 25.0},
                "Spreads": {"Jolly:0/32/0/0/0/32": 85.0},
                "Moves": {
                    "earthquake": 90.0,
                    "dragonclaw": 70.0,
                    "swordsdance": 55.0,
                    "protect": 60.0,
                },
            },
            "Amoonguss": {
                "usage": 0.18,
                "Raw count": 100000,
                "Abilities": {"regenerator": 100.0},
                "Items": {"sitrusberry": 40.0, "rockyhelmet": 30.0},
                "Spreads": {"Bold:32/0/20/0/12/0": 70.0},
                "Moves": {
                    "spore": 95.0,
                    "ragepowder": 85.0,
                    "pollenpuff": 50.0,
                    "protect": 60.0,
                },
            },
            "EmptyMon": {
                "usage": 0.05,
                "Raw count": 5000,
                "Abilities": {},
                "Items": {},
                "Spreads": {},
                "Moves": {},
            },
        }
    }


def _build_mock_gmm() -> dict[str, Any]:
    """Build a minimal GMM model dict for testing.

    Returns:
        Dict with "gmm" and "scaler" keys suitable for joblib dump.
    """
    rng = np.random.RandomState(42)
    gmm = GaussianMixture(n_components=2, random_state=42)
    scaler = StandardScaler()
    data = rng.randn(20, 4)
    scaler.fit(data)
    gmm.fit(scaler.transform(data))
    return {"gmm": gmm, "scaler": scaler}


def _make_cache(
    smogon_data: dict[str, Any] | None = None,
) -> tuple[TeambuildCache, str, str]:
    """Create a TeambuildCache backed by temporary JSON and GMM files.

    Args:
        smogon_data: Optional mock Smogon data dict. Uses 5-species mock if None.

    Returns:
        Tuple of (cache, tmp_smogon_path, tmp_gmm_path). Cleanup is the
        caller's responsibility.
    """
    if smogon_data is None:
        smogon_data = _build_mock_smogon_json()

    smogon_path = None
    gmm_path = None

    tmp_smogon = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    )
    json.dump(smogon_data, tmp_smogon)
    tmp_smogon.close()
    smogon_path = tmp_smogon.name

    tmp_gmm = tempfile.NamedTemporaryFile(
        mode="wb", suffix=".pkl", delete=False
    )
    joblib.dump(_build_mock_gmm(), tmp_gmm)
    tmp_gmm.close()
    gmm_path = tmp_gmm.name

    try:
        cache = load_teambuild_cache(
            smogon_path=smogon_path, gmm_path=gmm_path
        )
        return cache, smogon_path, gmm_path
    except Exception:
        os.unlink(smogon_path)
        os.unlink(gmm_path)
        raise


class TestTeambuildConfig:
    """Tests for the TeambuildConfig Pydantic model."""

    def test_default_values(self) -> None:
        config = TeambuildConfig()
        assert config.population_size == 100
        assert config.generations == 50
        assert config.mutation_rate == 0.05
        assert config.elite_fraction == 0.10

    def test_custom_values(self) -> None:
        config = TeambuildConfig(
            population_size=50,
            generations=20,
            mutation_rate=0.10,
            elite_fraction=0.15,
        )
        assert config.population_size == 50
        assert config.generations == 20
        assert config.mutation_rate == 0.10
        assert config.elite_fraction == 0.15

    def test_partial_override(self) -> None:
        config = TeambuildConfig(population_size=200)
        assert config.population_size == 200
        assert config.generations == 50

    def test_unknown_fields_filtered(self) -> None:
        config = TeambuildConfig.model_validate(
            {"population_size": 60, "extra_field": 999}
        )
        assert config.population_size == 60
        assert not hasattr(config, "extra_field")

    def test_type_validation_int(self) -> None:
        with pytest.raises(Exception):
            TeambuildConfig(population_size="not_an_int")  # type: ignore[arg-type]


class TestParseSpread:
    """Tests for the _parse_spread helper."""

    def test_standard_spread(self) -> None:
        nature, evs = _parse_spread("Jolly:0/32/0/0/0/32")
        assert nature == "Jolly"
        assert evs == [0, 32, 0, 0, 0, 32]

    def test_zero_spread(self) -> None:
        nature, evs = _parse_spread("Serious:0/0/0/0/0/0")
        assert nature == "Serious"
        assert evs == [0, 0, 0, 0, 0, 0]

    def test_partial_spread_missing_colon(self) -> None:
        nature, evs = _parse_spread("Adamant")
        assert nature == "Adamant"
        assert evs == [0, 0, 0, 0, 0, 0]

    def test_all_max_spread(self) -> None:
        nature, evs = _parse_spread("Modest:63/63/63/63/63/63")
        assert evs == [63, 63, 63, 63, 63, 63]


class TestWeightedChoice:
    """Tests for the _weighted_choice helper."""

    def test_returns_key_from_options(self) -> None:
        options = {"a": 1.0, "b": 0.0}
        result = _weighted_choice(options)
        assert result in options

    def test_empty_options_returns_empty(self) -> None:
        assert _weighted_choice({}) == ""

    def test_zero_weights_picks_random(self) -> None:
        options = {"x": 0.0, "y": 0.0, "z": 0.0}
        for _ in range(10):
            assert _weighted_choice(options) in options

    def test_probabilistic_distribution(self) -> None:
        options = {"dominant": 100.0, "rare": 0.1}
        results = [_weighted_choice(options) for _ in range(1000)]
        dominant_count = results.count("dominant")
        assert dominant_count > 900


class TestTopWeightedMoves:
    """Tests for the _top_weighted_moves helper."""

    def test_returns_top_n(self) -> None:
        pool = {"a": 10.0, "b": 5.0, "c": 1.0, "d": 0.5}
        result = _top_weighted_moves(pool, 2)
        assert result == ["a", "b"]

    def test_n_exceeds_pool_size(self) -> None:
        pool = {"a": 10.0, "b": 5.0}
        result = _top_weighted_moves(pool, 4)
        assert len(result) == 2
        assert set(result) == {"a", "b"}

    def test_empty_pool(self) -> None:
        assert _top_weighted_moves({}, 4) == []

    def test_single_entry(self) -> None:
        result = _top_weighted_moves({"only": 1.0}, 4)
        assert result == ["only"]


class TestTeambuildCache:
    """Integration tests for TeambuildCache with mock Smogon and GMM data."""

    def test_load_creates_cache_with_species_keys(self) -> None:
        cache, smogon_path, gmm_path = _make_cache()
        try:
            assert isinstance(cache, TeambuildCache)
            assert cache.n_species == 5
            assert len(cache.species_keys) == 5
        finally:
            os.unlink(smogon_path)
            os.unlink(gmm_path)

    def test_species_keys_sorted_by_usage_desc(self) -> None:
        cache, smogon_path, gmm_path = _make_cache()
        try:
            usages = [
                cache.get_entry(k).get("usage", 0)  # type: ignore[union-attr]
                for k in cache.species_keys
            ]
            for i in range(len(usages) - 1):
                assert usages[i] >= usages[i + 1], (
                    f"species_keys not sorted by usage descending: "
                    f"{cache.species_keys[i]}({usages[i]}) < "
                    f"{cache.species_keys[i + 1]}({usages[i + 1]})"
                )
        finally:
            os.unlink(smogon_path)
            os.unlink(gmm_path)

    def test_usage_weights_sum_to_one(self) -> None:
        cache, smogon_path, gmm_path = _make_cache()
        try:
            assert np.isclose(cache.usage_weights.sum(), 1.0, atol=0.001)
            assert np.all(cache.usage_weights >= 0)
        finally:
            os.unlink(smogon_path)
            os.unlink(gmm_path)

    def test_usage_weights_parallel_to_species_keys(self) -> None:
        cache, smogon_path, gmm_path = _make_cache()
        try:
            assert len(cache.usage_weights) == len(cache.species_keys)
        finally:
            os.unlink(smogon_path)
            os.unlink(gmm_path)

    def test_gmm_loaded(self) -> None:
        cache, smogon_path, gmm_path = _make_cache()
        try:
            assert cache.gmm is not None
            assert isinstance(cache.gmm, GaussianMixture)
            assert cache.gmm.n_components == 2
        finally:
            os.unlink(smogon_path)
            os.unlink(gmm_path)

    def test_scaler_loaded(self) -> None:
        cache, smogon_path, gmm_path = _make_cache()
        try:
            assert cache.scaler is not None
            assert isinstance(cache.scaler, StandardScaler)
        finally:
            os.unlink(smogon_path)
            os.unlink(gmm_path)

    def test_get_entry_by_lowercase(self) -> None:
        cache, smogon_path, gmm_path = _make_cache()
        try:
            entry = cache.get_entry("fluttermane")
            assert entry is not None
            assert entry.get("usage") == 0.55
        finally:
            os.unlink(smogon_path)
            os.unlink(gmm_path)

    def test_get_entry_by_title_case(self) -> None:
        cache, smogon_path, gmm_path = _make_cache()
        try:
            entry = cache.get_entry("FlutterMane")
            assert entry is not None
            assert entry.get("usage") == 0.55
        finally:
            os.unlink(smogon_path)
            os.unlink(gmm_path)

    def test_get_entry_unknown_species(self) -> None:
        cache, smogon_path, gmm_path = _make_cache()
        try:
            assert cache.get_entry("pikachu") is None
        finally:
            os.unlink(smogon_path)
            os.unlink(gmm_path)

    def test_sample_ability_returns_string(self) -> None:
        cache, smogon_path, gmm_path = _make_cache()
        try:
            ability = cache.sample_ability("fluttermane")
            assert isinstance(ability, str)
            assert ability in ("protosynthesis",)
        finally:
            os.unlink(smogon_path)
            os.unlink(gmm_path)

    def test_sample_ability_unknown_species(self) -> None:
        cache, smogon_path, gmm_path = _make_cache()
        try:
            assert cache.sample_ability("pikachu") == ""
        finally:
            os.unlink(smogon_path)
            os.unlink(gmm_path)

    def test_sample_ability_empty_abilities(self) -> None:
        cache, smogon_path, gmm_path = _make_cache()
        try:
            assert cache.sample_ability("emptymon") == ""
        finally:
            os.unlink(smogon_path)
            os.unlink(gmm_path)

    def test_sample_item_returns_string(self) -> None:
        cache, smogon_path, gmm_path = _make_cache()
        try:
            item = cache.sample_item("fluttermane")
            assert isinstance(item, str)
            assert item in ("boosterenergy", "focussash")
        finally:
            os.unlink(smogon_path)
            os.unlink(gmm_path)

    def test_sample_item_unknown_species(self) -> None:
        cache, smogon_path, gmm_path = _make_cache()
        try:
            assert cache.sample_item("pikachu") == ""
        finally:
            os.unlink(smogon_path)
            os.unlink(gmm_path)

    def test_sample_spread_returns_tuple(self) -> None:
        cache, smogon_path, gmm_path = _make_cache()
        try:
            nature, evs = cache.sample_spread("fluttermane")
            assert isinstance(nature, str)
            assert isinstance(evs, list)
            assert len(evs) == 6
            assert nature in ("Timid", "Modest")
        finally:
            os.unlink(smogon_path)
            os.unlink(gmm_path)

    def test_sample_spread_unknown_species(self) -> None:
        cache, smogon_path, gmm_path = _make_cache()
        try:
            nature, evs = cache.sample_spread("pikachu")
            assert nature == "serious"
            assert evs == [0, 0, 0, 0, 0, 0]
        finally:
            os.unlink(smogon_path)
            os.unlink(gmm_path)

    def test_sample_spread_empty_spreads(self) -> None:
        cache, smogon_path, gmm_path = _make_cache()
        try:
            nature, evs = cache.sample_spread("emptymon")
            assert nature == "serious"
            assert evs == [0, 0, 0, 0, 0, 0]
        finally:
            os.unlink(smogon_path)
            os.unlink(gmm_path)

    def test_sample_moves_returns_top_n(self) -> None:
        cache, smogon_path, gmm_path = _make_cache()
        try:
            moves = cache.sample_moves("fluttermane", n=3)
            assert len(moves) == 3
            assert moves[0] == "shadowball"
            assert moves[1] == "moonblast"
            assert moves[2] == "protect"
        finally:
            os.unlink(smogon_path)
            os.unlink(gmm_path)

    def test_sample_moves_default_n(self) -> None:
        cache, smogon_path, gmm_path = _make_cache()
        try:
            moves = cache.sample_moves("fluttermane")
            assert len(moves) == 4
        finally:
            os.unlink(smogon_path)
            os.unlink(gmm_path)

    def test_sample_moves_unknown_species(self) -> None:
        cache, smogon_path, gmm_path = _make_cache()
        try:
            assert cache.sample_moves("pikachu") == []
        finally:
            os.unlink(smogon_path)
            os.unlink(gmm_path)

    def test_sample_moves_empty_pool(self) -> None:
        cache, smogon_path, gmm_path = _make_cache()
        try:
            assert cache.sample_moves("emptymon") == []
        finally:
            os.unlink(smogon_path)
            os.unlink(gmm_path)

    def test_hydrate_pokemon_returns_full_pokemon(self) -> None:
        cache, smogon_path, gmm_path = _make_cache()
        try:
            mon = cache.hydrate_pokemon("fluttermane")
            assert isinstance(mon, Pokemon)
            assert mon.species == "fluttermane"
            assert mon.ability == "protosynthesis"
            assert mon.item != ""
            assert mon.nature != ""
            assert len(mon.moves) == 4
        finally:
            os.unlink(smogon_path)
            os.unlink(gmm_path)

    def test_hydrate_pokemon_unknown_species_fallback(self) -> None:
        cache, smogon_path, gmm_path = _make_cache()
        try:
            mon = cache.hydrate_pokemon("pikachu")
            assert isinstance(mon, Pokemon)
            assert mon.species == "pikachu"
        finally:
            os.unlink(smogon_path)
            os.unlink(gmm_path)

    def test_hydrate_team_returns_six_pokemon(self) -> None:
        cache, smogon_path, gmm_path = _make_cache()
        try:
            indices = [0, 1, 2, 3, 4, 1]
            team = cache.hydrate_team(indices)
            assert len(team) == 6
            assert all(isinstance(m, Pokemon) for m in team)
            assert team[0].species == cache.species_keys[0]
        finally:
            os.unlink(smogon_path)
            os.unlink(gmm_path)

    def test_team_to_showdown_format(self) -> None:
        cache, smogon_path, gmm_path = _make_cache()
        try:
            indices = [0, 1, 2, 3, 4, 1]
            team = cache.hydrate_team(indices)
            strategy = (0, 1, 2, 3, 4, 5)
            result = cache.team_to_showdown(team, strategy)
            assert isinstance(result, str)
            assert len(result) > 0
            assert "Ability:" in result
            assert "Level: 50" in result
        finally:
            os.unlink(smogon_path)
            os.unlink(gmm_path)

    def test_team_to_showdown_separates_by_double_newline(self) -> None:
        cache, smogon_path, gmm_path = _make_cache()
        try:
            indices = [0, 1, 2, 3, 4, 1]
            team = cache.hydrate_team(indices)
            strategy = (0, 1, 2, 3, 4, 5)
            result = cache.team_to_showdown(team, strategy)
            pokemon_blocks = result.split("\n\n")
            assert len(pokemon_blocks) == 6
        finally:
            os.unlink(smogon_path)
            os.unlink(gmm_path)

    def test_load_raises_on_missing_smogon(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_teambuild_cache(
                smogon_path="/nonexistent/path.json",
                gmm_path="models/archetype_gmm.pkl",
            )

    def test_load_raises_on_missing_gmm(self) -> None:
        smogon_data = _build_mock_smogon_json()
        tmp_smogon = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        json.dump(smogon_data, tmp_smogon)
        tmp_smogon.close()
        smogon_path = tmp_smogon.name
        try:
            with pytest.raises(FileNotFoundError):
                load_teambuild_cache(
                    smogon_path=smogon_path,
                    gmm_path="/nonexistent/model.pkl",
                )
        finally:
            os.unlink(smogon_path)
