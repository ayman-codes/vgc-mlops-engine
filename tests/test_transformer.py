import numpy as np
from src.agent.selection_policy.transformer import aggregate_macro_features, macro_features_array


def test_empty_team_returns_zeros() -> None:
    features = aggregate_macro_features([])
    assert features["avg_speed"] == np.float32(0.0)
    assert features["phys_spec_ratio"] == np.float32(0.0)
    assert features["bulk_index"] == np.float32(0.0)
    assert features["type_synergy_density"] == np.float32(0.0)


def test_single_known_pokemon_chart() -> None:
    features = aggregate_macro_features(["charizard"])
    assert features["avg_speed"] > np.float32(0.0)
    assert features["phys_spec_ratio"] > np.float32(0.0)
    assert features["bulk_index"] > np.float32(0.0)
    assert features["type_synergy_density"] >= np.float32(0.0)


def test_charizard_base_stats() -> None:
    features = aggregate_macro_features(["charizard"])
    assert features["avg_speed"] == np.float32(100.0)
    hp, df, spd = 78, 78, 85
    expected_bulk = (hp + df + spd) / 1
    assert features["bulk_index"] == np.float32(expected_bulk)


def test_clefable_is_specially_oriented() -> None:
    features = aggregate_macro_features(["clefable"])
    assert features["phys_spec_ratio"] < np.float32(1.0)


def test_stonjourner_is_physically_oriented() -> None:
    features = aggregate_macro_features(["stonjourner"])
    assert features["phys_spec_ratio"] > np.float32(1.0)


def test_team_macro_features_chart() -> None:
    team = ["charizard", "pikachu", "meowth", "eevee"]
    features = aggregate_macro_features(team)
    for key in ("avg_speed", "phys_spec_ratio", "bulk_index", "type_synergy_density"):
        val = features[key]
        assert isinstance(val, np.float32), f"{key} is {type(val)}, not float32"
        assert val >= np.float32(0.0), f"{key} is negative: {val}"


def test_output_no_one_hot_encoding() -> None:
    team = ["charizard", "blastoise", "venusaur", "pikachu", "eevee", "snorlax"]
    features = aggregate_macro_features(team)
    assert len(features) == 4
    for val in features.values():
        assert isinstance(val, np.float32)


def test_array_output_shape() -> None:
    team = ["charizard", "blastoise", "venusaur"]
    arr = macro_features_array(team)
    assert arr.shape == (4,)
    assert arr.dtype == np.float32


def test_unknown_species_skipped_gracefully() -> None:
    features = aggregate_macro_features(["not_a_real_pokemon_xyz"])
    assert features["avg_speed"] == np.float32(0.0)
    assert features["bulk_index"] == np.float32(0.0)


def test_team_with_unknown_species_produces_valid_features() -> None:
    features = aggregate_macro_features(["charizard", "aaaaa", "venusaur", "bbbbb"])
    assert features["avg_speed"] > np.float32(0.0)
    assert features["bulk_index"] > np.float32(0.0)


def test_team_of_six_produces_reasonable_values() -> None:
    team = ["charizard", "blastoise", "venusaur", "pikachu", "gengar", "snorlax"]
    features = aggregate_macro_features(team)
    assert 70.0 <= features["avg_speed"] <= 130.0
    assert 0.5 <= features["phys_spec_ratio"] <= 2.0
    assert 150.0 <= features["bulk_index"] <= 350.0
    assert 0.0 <= features["type_synergy_density"] <= 1.0
