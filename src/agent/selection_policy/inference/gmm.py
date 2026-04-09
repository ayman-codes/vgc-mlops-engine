import numpy as np
from typing import Dict, List, Any
from sklearn.mixture import GaussianMixture 

class ArchetypeGMM:
    def __init__(self, n_components: int = 4, covariance_type: str = 'full', random_state: int = 42) -> None:
        self.n_components = n_components
        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state
        )
        self.is_fitted = False

    def fit(self, training_data: np.ndarray[Any, Any]) -> None:
        """Fits the GMM to historical feature vectors."""
        self.model.fit(training_data)
        self.is_fitted = True

    def predict(self, feature_vectors: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Assigns cluster IDs to input vectors."""
        if not self.is_fitted:
            raise RuntimeError("GMM must be fitted before prediction.")
        return self.model.predict(feature_vectors) # type: ignore

    def extract_bayesian_parameters(
        self,
        training_data: np.ndarray[Any, Any],
        entity_labels: List[str]
    ) -> Dict[str, Any]:
        """
        Maps GMM clusters to Naive Bayes priors and likelihoods.
        training_data: Shape (N_samples, N_features)
        entity_labels: List of species names corresponding to each sample
        """
        if not self.is_fitted:
            raise RuntimeError("GMM must be fitted before parameter extraction.")

        labels = self.predict(training_data)
        n_samples = len(labels)
        
        priors: Dict[str, float] = {}
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            priors[f"archetype_{label}"] = float(count) / n_samples

        likelihoods: Dict[str, Dict[str, float]] = {f"archetype_{label}": {} for label in unique_labels}
        cluster_totals = {label: 0 for label in unique_labels}

        for label, entity in zip(labels, entity_labels):
            arch_key = f"archetype_{label}"
            likelihoods[arch_key][entity] = likelihoods[arch_key].get(entity, 0.0) + 1.0
            cluster_totals[label] += 1

        for label in unique_labels:
            arch_key = f"archetype_{label}"
            total = cluster_totals[label]
            if total > 0:
                for entity in likelihoods[arch_key]:
                    likelihoods[arch_key][entity] /= float(total)

        return {
            "priors": priors,
            "likelihoods": likelihoods
        }