"""Archetype classification using a pre-trained GMM model.

Loads the GMM + StandardScaler from models/archetype_gmm.pkl
and provides probability distribution and hard-label prediction.
"""

import os
from typing import Any
import numpy as np
from numpy.typing import NDArray
import joblib
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.agent.selection_policy.transformer import macro_features_array

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "archetype_gmm.pkl")

_gmm: GaussianMixture | None = None
_scaler: StandardScaler | None = None


def _load_model() -> tuple[GaussianMixture, StandardScaler | None]:
    global _gmm, _scaler
    if _gmm is None:
        loaded = joblib.load(MODEL_PATH)
        if isinstance(loaded, dict):
            _gmm = loaded["gmm"]
            _scaler = loaded.get("scaler", None)
        else:
            _gmm = loaded
            _scaler = None
    return _gmm, _scaler


def predict_archetype(opponent_species_list: list[str]) -> NDArray[np.float64]:
    """Predict the archetype probability distribution for an opponent team.

    Args:
        opponent_species_list: List of opponent species names (lowercase).

    Returns:
        Float64 array of shape (n_components,) where each entry is the
        probability that the team belongs to that archetype cluster.
    """
    model, scaler = _load_model()
    features = macro_features_array(opponent_species_list).reshape(1, -1)
    if scaler is not None:
        features = scaler.transform(features)
    probs: NDArray[Any] = model.predict_proba(features)
    return probs.flatten()


def predict_archetype_label(opponent_species_list: list[str]) -> int:
    """Predict the hard archetype label for an opponent team.

    Args:
        opponent_species_list: List of opponent species names (lowercase).

    Returns:
        Integer cluster index (0 or 1 for the 2-cluster GMM).
    """
    model, scaler = _load_model()
    features = macro_features_array(opponent_species_list).reshape(1, -1)
    if scaler is not None:
        features = scaler.transform(features)
    return int(model.predict(features)[0])
