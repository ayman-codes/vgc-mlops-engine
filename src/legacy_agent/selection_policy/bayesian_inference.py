import json
import random
from typing import Dict, Any, List

PRIORS_PATH = "data/processed/bayesian_priors.json"

class BayesianInferenceEngine:
    """
    Hydrates opponent Team Preview data dynamically using Bayesian priors.
    Applies Procedural Variance Injection (20% Rule).
    """
    def __init__(self, priors_path: str = PRIORS_PATH) -> None:
        self.priors: Dict[str, Any] = {}
        try:
            with open(priors_path, "r", encoding="utf-8") as f:
                self.priors = json.load(f)
        except FileNotFoundError:
            pass

    def _get_most_likely(self, species: str, category: str, k: int = 1) -> List[str]:
        if species not in self.priors or category not in self.priors[species]:
            return []
        
        counts = self.priors[species][category]
        if not counts:
            return []

        sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_items[:k]]

    def predict_pokemon_build(self, species: str) -> Dict[str, Any]:
        """
        Predicts the optimal build, applying 20% procedural variance to prevent matrix solver brittleness.
        """
        build: Dict[str, Any] = {
            "species": species,
            "item": None,
            "ability": None,
            "moves": []
        }
        
        likely_items = self._get_most_likely(species, "items", k=3)
        if likely_items:
            build["item"] = likely_items[0] if random.random() > 0.2 else random.choice(likely_items)
            
        likely_abilities = self._get_most_likely(species, "abilities", k=2)
        if likely_abilities:
            build["ability"] = likely_abilities[0] if random.random() > 0.2 else random.choice(likely_abilities)

        likely_moves = self._get_most_likely(species, "moves", k=6)
        if likely_moves:
            if random.random() > 0.2:
                build["moves"] = likely_moves[:4]
            else:
                # 20% variance: randomly sample 4 from top 6
                build["moves"] = random.sample(likely_moves, min(4, len(likely_moves)))

        return build

    def hydrate_team(self, species_list: List[str]) -> Dict[str, Dict[str, Any]]:
        return {species: self.predict_pokemon_build(species) for species in species_list}
