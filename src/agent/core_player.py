from typing import Any
import numpy as np

from poke_env.player import Player
from poke_env.battle import AbstractBattle

from src.agent.selection_policy.classifier import predict_archetype
from src.agent.selection_policy.hydrator import hydrate_team
from src.agent.selection_policy.double_oracle import (
    enumerate_top_k_strategies,
)
from src.agent.selection_policy.optimizer import nash_loop
from src.agent.selection_policy.utils import build_team_order, shannon_entropy
from src.config.selection_model import SelectionConfig


class MyVGCAgent(Player):
    """Core VGC Agent combining Battle Policy and Selection Policy via poke-env.

    The teampreview hook runs the full V2 4-stage selection pipeline:
    Classifier -> Hydrator -> Double Oracle -> Nash Optimizer.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._selection_config = SelectionConfig()

    def choose_move(self, battle: AbstractBattle) -> Any:
        """Battle Policy entrypoint."""
        return self.choose_random_move(battle)

    async def teampreview(self, battle: AbstractBattle) -> str:
        """Selection Policy (Team Preview) entrypoint.

        Runs the V2 4-stage selection pipeline:
        1. Classifier: predict opponent archetype distribution via GMM
        2. Hydrator: build fully hydrated Pokemon from species names
        3. Double Oracle: enumerate top-K lead strategies
        4. Nash Optimizer: resolve mixed-strategy equilibrium

        Falls back to random teampreview on any pipeline failure.
        """
        our_species = [mon.species for mon in battle.team.values()]
        opponent_species = [mon.species for mon in battle.opponent_team.values()]

        import mlflow
        try:
            with mlflow.start_run(nested=True, run_name="teampreview"):
                mlflow.log_params({
                    "our_species": ",".join(our_species),
                    "opponent_species": ",".join(opponent_species),
                })

                archetype_dist = predict_archetype(opponent_species)
                mlflow.log_metric("archetype_peak", float(archetype_dist.max()))
                mlflow.log_metric("archetype_entropy", shannon_entropy(archetype_dist))

                our_team = hydrate_team(
                    our_species,
                    archetype_distribution=archetype_dist.tolist(),
                    variance=self._selection_config.procedural_variance,
                )
                opponent_team = hydrate_team(
                    opponent_species,
                    archetype_distribution=archetype_dist.tolist(),
                    variance=self._selection_config.procedural_variance,
                )

                our_strategies = enumerate_top_k_strategies(our_team, k=3, opponent_team=opponent_team)
                opp_strategies = enumerate_top_k_strategies(opponent_team, k=3, opponent_team=our_team)

                if not our_strategies or not opp_strategies:
                    return self.random_teampreview(battle)

                final_mix, final_strategies = await nash_loop(
                    our_team,
                    opponent_team,
                    k_start=3,
                    k_max=min(self._selection_config.async_batch_size + 2, 15),
                    timeout_sec=self._selection_config.timeout_limit_sec,
                )

                mlflow.log_metric("final_entropy", shannon_entropy(final_mix))

                if not final_strategies:
                    return self.random_teampreview(battle)

                best_idx = int(np.argmax(final_mix[: len(final_strategies)]))
                best_strategy = final_strategies[best_idx]
                return build_team_order(our_species, best_strategy)
        except Exception as exc:
            try:
                import mlflow
                mlflow.log_metric("pipeline_error", 1.0)
                mlflow.log_param("pipeline_exception", str(exc)[:250])
            except Exception:
                pass
            return self.random_teampreview(battle)
