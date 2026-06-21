import pytest
import numpy as np
from src.agent.selection_policy.inference.gmm import ArchetypeGMM

def test_gmm_fit_predict() -> None:
    gmm = ArchetypeGMM(n_components=2, random_state=42)
    data = np.array([[0.1, 0.2], [0.15, 0.18], [0.9, 0.8], [0.95, 0.85]])
    
    gmm.fit(data)
    assert gmm.is_fitted is True
    
    predictions = gmm.predict(data)
    assert len(predictions) == 4
    assert predictions[0] == predictions[1]
    assert predictions[2] == predictions[3]
    assert predictions[0] != predictions[2]

def test_gmm_extract_bayesian_parameters() -> None:
    gmm = ArchetypeGMM(n_components=2, random_state=42)
    data = np.array([[0.1], [0.2], [0.9], [0.8]])
    entities = ["Incineroar", "Incineroar", "Rillaboom", "Rillaboom"]
    
    gmm.fit(data)
    params = gmm.extract_bayesian_parameters(data, entities)
    
    assert "priors" in params
    assert "likelihoods" in params
    assert len(params["priors"]) == 2
    assert sum(params["priors"].values()) == pytest.approx(1.0)
    
    arch_keys = list(params["priors"].keys())
    assert "Incineroar" in params["likelihoods"][arch_keys[0]] or "Rillaboom" in params["likelihoods"][arch_keys[0]]

def test_gmm_unfitted_error() -> None:
    gmm = ArchetypeGMM()
    with pytest.raises(RuntimeError):
        gmm.predict(np.array([[1.0, 2.0]]))
        
    with pytest.raises(RuntimeError):
        gmm.extract_bayesian_parameters(np.array([[1.0, 2.0]]), ["Pikachu"])

if __name__ == "__main__":
    pytest.main([__file__])