import pytest
from unittest.mock import MagicMock
from src.agent.selection_policy.inference.bayesian import BayesianHiddenStatePredictor

def test_compute_posterior_normalization() -> None:
    predictor = BayesianHiddenStatePredictor(smoothing_factor=0.01)
    priors = {"Fast_Physical": 0.7, "Bulky_Support": 0.3}
    likelihoods = {
        "Fast_Physical": {"Incineroar": 0.8, "Rillaboom": 0.9},
        "Bulky_Support": {"Incineroar": 0.1, "Rillaboom": 0.2}
    }
    context = ["Incineroar", "Rillaboom"]
    
    posteriors = predictor.compute_posterior(priors, likelihoods, context)
    
    assert "Fast_Physical" in posteriors
    assert "Bulky_Support" in posteriors
    assert posteriors["Fast_Physical"] > posteriors["Bulky_Support"]
    assert pytest.approx(sum(posteriors.values())) == 1.0

def test_compute_posterior_zero_evidence_fallback() -> None:
    predictor = BayesianHiddenStatePredictor(smoothing_factor=0.0)
    priors = {"A": 0.5, "B": 0.5}
    likelihoods = {
        "A": {"Unknown": 0.0},
        "B": {"Unknown": 0.0}
    }
    context = ["Unknown"]
    
    posteriors = predictor.compute_posterior(priors, likelihoods, context)
    
    assert posteriors["A"] == 0.5
    assert posteriors["B"] == 0.5

def test_infer_archetype_probabilities() -> None:
    predictor = BayesianHiddenStatePredictor()
    
    target_view = MagicMock()
    target_view.species.name = "Target"
    
    ally_view = MagicMock()
    ally_view.species.name = "Ally1"
    
    usage_data = {
        "priors": {"build1": 0.5, "build2": 0.5},
        "likelihoods": {
            "build1": {"Ally1": 0.9},
            "build2": {"Ally1": 0.1}
        }
    }
    
    probs = predictor.infer_archetype_probabilities(target_view, [target_view, ally_view], usage_data)
    
    assert probs["build1"] == 0.9
    assert probs["build2"] == 0.1

if __name__ == "__main__":
    pytest.main([__file__])