from typing import Dict, List, Any
from vgc2.battle_engine.view import PokemonView

class BayesianHiddenStatePredictor:
    def __init__(self, smoothing_factor: float = 1e-4) -> None:
        self.smoothing_factor = smoothing_factor

    def compute_posterior(
        self, 
        priors: Dict[str, float], 
        likelihoods: Dict[str, Dict[str, float]], 
        context_entities: List[str]
    ) -> Dict[str, float]:
        posteriors: Dict[str, float] = {}
        evidence: float = 0.0

        for state, prior in priors.items():
            likelihood = 1.0
            state_likelihoods = likelihoods.get(state, {})
            
            for entity in context_entities:
                prob = state_likelihoods.get(entity, self.smoothing_factor)
                likelihood *= float(prob)
            
            unnormalized_posterior = float(prior) * likelihood
            posteriors[state] = unnormalized_posterior
            evidence += unnormalized_posterior

        if evidence <= 0.0:
            num_states = len(priors)
            if num_states == 0:
                return {}
            return {state: 1.0 / float(num_states) for state in priors}

        return {state: val / evidence for state, val in posteriors.items()}
        
    def infer_archetype_probabilities(
        self, 
        target_view: PokemonView, 
        team_views: List[PokemonView], 
        usage_data: Dict[str, Any]
    ) -> Dict[str, float]:
        if not usage_data or "priors" not in usage_data or "likelihoods" not in usage_data:
            return {}
            
        target_species = str(target_view.species.name) if target_view.species else ""
        if not target_species:
            return {}

        context_entities: List[str] = []
        for v in team_views:
            if v.species and v.species.name != target_species:
                context_entities.append(str(v.species.name))
        
        priors: Dict[str, float] = usage_data["priors"]
        likelihoods: Dict[str, Dict[str, float]] = usage_data["likelihoods"]
        
        return self.compute_posterior(priors, likelihoods, context_entities)