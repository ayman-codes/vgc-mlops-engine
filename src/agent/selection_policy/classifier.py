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
    model, scaler = _load_model()
    features = macro_features_array(opponent_species_list).reshape(1, -1)
    if scaler is not None:
        features = scaler.transform(features)
    probs: NDArray[Any] = model.predict_proba(features)
    return probs.flatten()


def predict_archetype_label(opponent_species_list: list[str]) -> int:
    model, scaler = _load_model()
    features = macro_features_array(opponent_species_list).reshape(1, -1)
    if scaler is not None:
        features = scaler.transform(features)
    return int(model.predict(features)[0])
