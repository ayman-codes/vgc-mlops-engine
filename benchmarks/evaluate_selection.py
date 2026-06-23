import asyncio
import time
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import os
from typing import Any

from poke_env import cross_evaluate
from poke_env.player import RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client import ServerConfiguration
from poke_env.battle import AbstractBattle
from poke_env.teambuilder import ConstantTeambuilder

from src.agent.selection_policy.classifier import predict_archetype
from src.agent.selection_policy.hydrator import hydrate_team
from src.agent.selection_policy.double_oracle import (
    enumerate_top_k_strategies,
)
from src.agent.selection_policy.optimizer import nash_loop
from src.agent.selection_policy.utils import build_team_order, shannon_entropy
from src.config.selection_model import SelectionConfig


class V2SelectionBenchmarkPlayer(SimpleHeuristicsPlayer):  # type: ignore[misc]
    """Benchmark player using SimpleHeuristics for combat and V2
    selection pipeline for Team Preview.

    Inherits battle logic from SimpleHeuristicsPlayer and overrides
    teampreview with the full V2 4-stage pipeline:
    Classifier -> Hydrator -> Double Oracle -> Nash Optimizer.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._config = SelectionConfig()
        self._all_entropies: list[float] = []
        self._all_matrix_sizes: list[tuple[int, int]] = []
        self._all_resolution_ms: list[float] = []

    async def teampreview(self, battle: AbstractBattle) -> str:
        """Run the V2 selection pipeline and return optimal team order."""
        start_time = time.time()

        our_species = [mon.species for mon in battle.team.values()]
        opp_species = [mon.species for mon in battle.opponent_team.values()]

        try:
            archetype_dist = predict_archetype(opp_species)

            our_team = hydrate_team(
                our_species,
                archetype_distribution=archetype_dist.tolist(),
                variance=self._config.procedural_variance,
            )
            opponent_team = hydrate_team(
                opp_species,
                archetype_distribution=archetype_dist.tolist(),
                variance=self._config.procedural_variance,
            )

            our_strategies = enumerate_top_k_strategies(our_team, k=3, opponent_team=opponent_team)
            opp_strategies = enumerate_top_k_strategies(opponent_team, k=3, opponent_team=our_team)

            if not our_strategies or not opp_strategies:
                self._all_resolution_ms.append((time.time() - start_time) * 1000)
                return self.random_teampreview(battle)

            final_mix, final_strategies = await nash_loop(
                our_team,
                opponent_team,
                k_start=3,
                k_max=min(self._config.async_batch_size + 2, 15),
                timeout_sec=self._config.timeout_limit_sec,
            )

            self._all_resolution_ms.append((time.time() - start_time) * 1000)
            self._all_entropies.append(shannon_entropy(final_mix))
            self._all_matrix_sizes.append((
                len(final_strategies),
                len(final_strategies),
            ))

            if not final_strategies:
                return self.random_teampreview(battle)

            best_idx = int(np.argmax(final_mix[: len(final_strategies)]))
            best_strategy = final_strategies[best_idx]
            return build_team_order(our_species, best_strategy)
        except Exception:
            self._all_resolution_ms.append((time.time() - start_time) * 1000)
            return self.random_teampreview(battle)


async def run_benchmark() -> None:
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Selection_Policy_Benchmark")

    with mlflow.start_run():
        print("Starting V2 Selection Policy Cross-Evaluation Tournament...")

        battle_format = "gen9customgame"

        server_config = ServerConfiguration(
            "ws://localhost:8000/showdown/websocket",
            "https://play.pokemonshowdown.com/action.php?",
        )

        DUMMY_TEAM = """
Incineroar @ Sitrus Berry
Ability: Intimidate
Level: 50
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Flare Blitz
- Knock Off
- Parting Shot
- Fake Out

Rillaboom @ Assault Vest
Ability: Grassy Surge
Level: 50
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Wood Hammer
- Grassy Glide
- U-turn
- Fake Out

Flutter Mane @ Booster Energy
Ability: Protosynthesis
Level: 50
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Moonblast
- Shadow Ball
- Dazzling Gleam
- Protect

Urshifu-Rapid-Strike @ Mystic Water
Ability: Unseen Fist
Level: 50
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Surging Strikes
- Close Combat
- Aqua Jet
- Protect

Ogerpon-Wellspring @ Wellspring Mask
Ability: Water Absorb
Level: 50
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Ivy Cudgel
- Horn Leech
- Spiky Shield
- Follow Me

Chien-Pao @ Focus Sash
Ability: Sword of Ruin
Level: 50
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Icicle Crash
- Sucker Punch
- Sacred Sword
- Protect
"""
        team_builder = ConstantTeambuilder(DUMMY_TEAM)

        v2_selection_player = V2SelectionBenchmarkPlayer(
            battle_format=battle_format,
            server_configuration=server_config,
            team=team_builder,
            max_concurrent_battles=10,
        )

        players = [
            RandomPlayer(
                battle_format=battle_format,
                server_configuration=server_config,
                team=team_builder,
                max_concurrent_battles=10,
            ),
            MaxBasePowerPlayer(
                battle_format=battle_format,
                server_configuration=server_config,
                team=team_builder,
                max_concurrent_battles=10,
            ),
            SimpleHeuristicsPlayer(
                battle_format=battle_format,
                server_configuration=server_config,
                team=team_builder,
                max_concurrent_battles=10,
            ),
            v2_selection_player,
        ]

        n_challenges = 10
        cross_eval_results = await cross_evaluate(players, n_challenges=n_challenges)

        player_names = ["Random", "MaxBasePower", "SimpleHeuristics", "V2Selection"]
        win_rates = []

        for i, p1 in enumerate(players):
            total_wins = 0
            total_matches = 0
            for j, p2 in enumerate(players):
                if i != j:
                    wins = cross_eval_results.get(p1.username, {}).get(p2.username, 0)
                    total_wins += wins if wins is not None else 0
                    total_matches += n_challenges

            win_rate = total_wins / total_matches if total_matches > 0 else 0
            win_rates.append(win_rate)
            mlflow.log_metric(f"{player_names[i]}_win_rate", win_rate)
            print(f"{player_names[i]} Win Rate: {win_rate:.2%}")

        if v2_selection_player._all_entropies:
            mlflow.log_metric(
                "v2_selection_entropy_mean",
                sum(v2_selection_player._all_entropies) / len(v2_selection_player._all_entropies),
            )
            mlflow.log_metric(
                "v2_selection_entropy_count",
                len(v2_selection_player._all_entropies),
            )
        if v2_selection_player._all_resolution_ms:
            mlflow.log_metric(
                "v2_resolution_time_ms_mean",
                sum(v2_selection_player._all_resolution_ms) / len(v2_selection_player._all_resolution_ms),
            )
            mlflow.log_metric(
                "v2_resolution_time_ms_max",
                max(v2_selection_player._all_resolution_ms),
            )
        if v2_selection_player._all_matrix_sizes:
            avg_m = sum(s[0] for s in v2_selection_player._all_matrix_sizes) / len(v2_selection_player._all_matrix_sizes)
            avg_n = sum(s[1] for s in v2_selection_player._all_matrix_sizes) / len(v2_selection_player._all_matrix_sizes)
            mlflow.log_metric("v2_matrix_size_m_mean", avg_m)
            mlflow.log_metric("v2_matrix_size_n_mean", avg_n)

        plt.figure(figsize=(10, 6))
        plt.bar(
            player_names,
            [wr * 100 for wr in win_rates],
            color=["gray", "orange", "blue", "green"],
        )
        plt.title("Selection Policy Cross-Evaluation Win Rates (V2 Pipeline)")
        plt.ylabel("Win Rate (%)")
        plt.ylim(0, 100)

        for i, wr in enumerate(win_rates):
            plt.text(i, wr * 100 + 2, f"{wr * 100:.1f}%", ha="center")

        os.makedirs("images", exist_ok=True)
        img_path = "images/selection_benchmarks.png"
        plt.savefig(img_path)
        mlflow.log_artifact(img_path)
        print(f"Chart saved to {img_path}")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
