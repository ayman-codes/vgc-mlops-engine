import numpy as np
from src.agent.selection_policy.classifier import predict_archetype, predict_archetype_label


def test_predict_archetype_returns_probability_vector() -> None:
    team = ["charizard", "blastoise", "venusaur", "pikachu", "gengar", "snorlax"]
    probs = predict_archetype(team)
    assert isinstance(probs, np.ndarray)
    assert probs.ndim == 1


def test_predict_archetype_probabilities_sum_to_one() -> None:
    team = ["charizard", "blastoise", "venusaur", "pikachu", "gengar", "snorlax"]
    probs = predict_archetype(team)
    assert np.isclose(np.sum(probs), 1.0, rtol=1e-5)


def test_predict_archetype_all_positive() -> None:
    team = ["charizard", "blastoise", "venusaur"]
    probs = predict_archetype(team)
    assert np.all(probs >= 0.0)


def test_predict_archetype_different_teams_different_results() -> None:
    team_a = ["charizard", "blastoise", "venusaur"]
    team_b = ["diglett", "stonjourner", "conkeldurr"]
    probs_a = predict_archetype(team_a)
    probs_b = predict_archetype(team_b)
    assert not np.array_equal(probs_a, probs_b)


def test_predict_archetype_label_is_valid_index() -> None:
    team = ["charizard", "blastoise", "venusaur"]
    label = predict_archetype_label(team)
    probs = predict_archetype(team)
    assert 0 <= label < len(probs)


def test_predict_archetype_label_consistent_with_argmax() -> None:
    team = ["pikachu", "eevee", "meowth"]
    label = predict_archetype_label(team)
    probs = predict_archetype(team)
    assert label == int(np.argmax(probs))


def test_predict_archetype_single_pokemon() -> None:
    probs = predict_archetype(["charizard"])
    assert np.isclose(np.sum(probs), 1.0, rtol=1e-5)
    assert len(probs) == 2


def test_predict_archetype_empty_team() -> None:
    probs = predict_archetype([])
    assert np.isclose(np.sum(probs), 1.0, rtol=1e-5)
    assert np.all(probs >= 0.0)


def test_predict_archetype_with_unknown_species() -> None:
    team = ["not_a_real_pokemon", "also_not_real"]
    probs = predict_archetype(team)
    assert np.isclose(np.sum(probs), 1.0, rtol=1e-5)


def test_predict_archetype_bulk_team() -> None:
    team = ["snorlax", "blissey", "toxapex", "ferrothorn", "clefable", "corviknight"]
    probs = predict_archetype(team)
    assert np.isclose(np.sum(probs), 1.0, rtol=1e-5)


def test_predict_archetype_speed_team() -> None:
    team = ["regieleki", "dragapult", "greninja", "talonflame", "accelgor", "ninjask"]
    probs = predict_archetype(team)
    assert np.isclose(np.sum(probs), 1.0, rtol=1e-5)
