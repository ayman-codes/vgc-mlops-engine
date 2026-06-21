from src.agent.selection_policy.bayesian_inference import BayesianInferenceEngine

import pathlib

def test_bayesian_inference_hydration(tmp_path: pathlib.Path) -> None:
    import json
    
    # Mock priors
    priors_file = tmp_path / "mock_priors.json"
    mock_data = {
        "Incineroar": {
            "items": {"Sitrus Berry": 50, "Safety Goggles": 10},
            "abilities": {"Intimidate": 60},
            "moves": {"Fake Out": 60, "Parting Shot": 55, "Knock Off": 50, "Flare Blitz": 40},
            "total": 60
        }
    }
    with open(priors_file, "w", encoding="utf-8") as f:
        json.dump(mock_data, f)
        
    engine = BayesianInferenceEngine(priors_path=str(priors_file))
    
    hydrated = engine.hydrate_team(["Incineroar"])
    assert "Incineroar" in hydrated
    build = hydrated["Incineroar"]
    
    assert build["species"] == "Incineroar"
    assert build["item"] in ["Sitrus Berry", "Safety Goggles"]
    assert build["ability"] == "Intimidate"
    # Given procedural variance, moves might be 4 out of 4
    assert len(build["moves"]) <= 4

def test_procedural_variance() -> None:
    import json
    
    # Mock priors
    import tempfile
    import os
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
        mock_data = {
            "Pikachu": {
                "items": {"Light Ball": 100},
                "abilities": {"Static": 100},
                "moves": {"Thunderbolt": 100, "Volt Tackle": 100, "Protect": 100, "Fake Out": 100, "Nuzzle": 100, "Iron Tail": 100},
                "total": 100
            }
        }
        json.dump(mock_data, f)
        temp_name = f.name
        
    engine = BayesianInferenceEngine(priors_path=temp_name)
    
    # Run multiple times to observe variance
    builds = [engine.predict_pokemon_build("Pikachu") for _ in range(100)]
    
    moves_sets = [tuple(sorted(b["moves"])) for b in builds]
    unique_move_sets = set(moves_sets)
    
    # 20% variance should yield more than 1 unique move set
    assert len(unique_move_sets) > 1
    os.remove(temp_name)
