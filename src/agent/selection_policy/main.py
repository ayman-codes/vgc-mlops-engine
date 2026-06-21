"""Selection Policy Player — teampreview hook that runs the full pipeline.

Classifier → Hydrator → Double Oracle → Optimizer → team order output.
"""

from typing import Any
import numpy as np
from numpy.typing import NDArray

from poke_env.player import Player

from src.agent.base import Pokemon
from src.agent.selection_policy.classifier import predict_archetype
from src.agent.selection_policy.hydrator import hydrate_team
from src.agent.selection_policy.double_oracle import (
    Strategy,
    enumerate_top_k_strategies,
)
from src.agent.selection_policy.optimizer import nash_loop
from src.config.selection_model import SelectionConfig

_SPECIES_CACHE: dict[str, str] | None = None


def _species_showdown_name(species: str) -> str:
    global _SPECIES_CACHE
    if _SPECIES_CACHE is None:
        from poke_env.data import GenData
        _SPECIES_CACHE = {}
        gd = GenData.from_gen(9)
        for pname, entry in gd.pokedex.items():
            _SPECIES_CACHE[pname] = entry.get("name", pname)
    return _SPECIES_CACHE.get(species, species)


def _build_team_order(team: list[Pokemon], strategy: Strategy) -> str:
    species = [team[i].species for i in strategy]
    showdown_names = [_species_showdown_name(s) for s in species]
    return "/team " + "|".join(showdown_names)


def _shannon_entropy(probs: NDArray[Any]) -> float:
    p = probs[probs > 0]
    return float(-np.sum(p * np.log2(p)))


class SelectionPolicyPlayer(Player):
    """A Player that uses the policy selection pipeline to choose its team order.

    Args:
        team_species: List of 6 species names (lowercase) for our team.
        config: SelectionConfig controlling variance, timeout, and batch size.
    """

    def __init__(
        self,
        team_species: list[str],
        config: SelectionConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._team_species = team_species
        self._config = config or SelectionConfig()

    async def teampreview(self, battle: Any) -> str:
        opponent_species = [m.species for m in battle.opponent_team]

        import mlflow

        with mlflow.start_run(nested=True, run_name="teampreview"):
            mlflow.log_params({
                "opponent_species": ",".join(opponent_species),
                "procedural_variance": self._config.procedural_variance,
                "timeout_limit_sec": self._config.timeout_limit_sec,
                "async_batch_size": self._config.async_batch_size,
            })

            archetype_dist = predict_archetype(opponent_species)
            mlflow.log_metric("archetype_peak", float(archetype_dist.max()))
            mlflow.log_metric("archetype_entropy", _shannon_entropy(archetype_dist))

            our_team = hydrate_team(
                self._team_species,
                archetype_distribution=archetype_dist.tolist(),
                variance=self._config.procedural_variance,
            )
            opponent_team = hydrate_team(
                opponent_species,
                archetype_distribution=archetype_dist.tolist(),
                variance=self._config.procedural_variance,
            )

            our_strategies = enumerate_top_k_strategies(our_team, k=3)
            opp_strategies = enumerate_top_k_strategies(opponent_team, k=3)

            if not our_strategies or not opp_strategies:
                return self.random_teampreview(battle)

            try:
                final_mix = await nash_loop(
                    our_team,
                    opponent_team,
                    k_start=3,
                    k_max=min(
                        self._config.async_batch_size + 2,
                        15,
                    ),
                    timeout_sec=self._config.timeout_limit_sec,
                )

                mlflow.log_metric("final_entropy", _shannon_entropy(final_mix))

                available = len(our_strategies)
                valid_len = min(available, len(final_mix))
                best_idx = int(np.argmax(final_mix[:valid_len]))
                best_strategy = our_strategies[best_idx]
                return _build_team_order(our_team, best_strategy)
            except Exception:
                return self.random_teampreview(battle)
