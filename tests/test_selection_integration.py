"""End-to-end integration tests for the V2 Selection Policy pipeline.

Tests the full 4-stage pipeline:
Classifier -> Hydrator -> Double Oracle -> Nash Optimizer -> team string output.
Mocks only the async battle function; all other components run against real data.
"""

import asyncio

import numpy as np
import pytest

from src.agent.base import Pokemon, Move
from src.agent.selection_policy.classifier import predict_archetype, predict_archetype_label
from src.agent.selection_policy.hydrator import hydrate_team
from src.agent.selection_policy.double_oracle import (
    Strategy,
    enumerate_top_k_strategies,
)
from src.agent.selection_policy.optimizer import nash_loop
from src.agent.selection_policy.utils import build_team_order, shannon_entropy
from src.config.selection_model import SelectionConfig


def _make_diverse_team() -> list[Pokemon]:
    """Build a team of 6 known species with diverse types.

    Uses species that exist in both GenData and the Smogon usage data.
    """
    return [
        Pokemon(
            species="snorlax",
            moves=[Move(name="bodyslam"), Move(name="earthquake"),
                   Move(name="icepunch"), Move(name="protect")],
            nature="adamant", ev_hp=252, ev_atk=252,
        ),
        Pokemon(
            species="charizard",
            moves=[Move(name="heatwave"), Move(name="airslash"),
                   Move(name="solarbeam"), Move(name="protect")],
            nature="timid", ev_hp=4, ev_spa=252, ev_spe=252,
        ),
        Pokemon(
            species="gyarados",
            moves=[Move(name="waterfall"), Move(name="icefang"),
                   Move(name="dragondance"), Move(name="protect")],
            nature="jolly", ev_hp=4, ev_atk=252, ev_spe=252,
        ),
        Pokemon(
            species="gengar",
            moves=[Move(name="shadowball"), Move(name="sludgebomb"),
                   Move(name="dazzlinggleam"), Move(name="protect")],
            nature="timid", ev_hp=4, ev_spa=252, ev_spe=252,
        ),
        Pokemon(
            species="alakazam",
            moves=[Move(name="psychic"), Move(name="focusblast"),
                   Move(name="shadowball"), Move(name="protect")],
            nature="timid", ev_hp=4, ev_spa=252, ev_spe=252,
        ),
        Pokemon(
            species="dragonite",
            moves=[Move(name="outrage"), Move(name="extremespeed"),
                   Move(name="earthquake"), Move(name="dragondance")],
            nature="adamant", ev_hp=4, ev_atk=252, ev_spe=252,
        ),
    ]


async def _mock_battle(
    our_team: list[Pokemon],
    our_strategy: Strategy,
    opponent_team: list[Pokemon],
    opponent_strategy: Strategy,
) -> float:
    """Mock async battle that returns win based on lead species index sum."""
    our_score = float(our_strategy[0] + our_strategy[1])
    opp_score = float(opponent_strategy[0] + opponent_strategy[1])
    return 1.0 if our_score > opp_score else 0.0


class TestSelectionPipelineIntegration:
    """Integration tests for the full V2 selection pipeline."""

    def test_full_pipeline_runs_without_errors(self) -> None:
        """The complete 4-stage pipeline runs end-to-end without exceptions."""
        team = _make_diverse_team()
        opp_team = _make_diverse_team()

        our_species = [mon.species for mon in team]
        opp_species = [mon.species for mon in opp_team]

        async def _run() -> None:
            archetype_dist = predict_archetype(opp_species)
            assert archetype_dist.shape == (2,)
            assert np.isclose(archetype_dist.sum(), 1.0, atol=0.01)

            label = predict_archetype_label(opp_species)
            assert label in (0, 1)

            our_hydrated = hydrate_team(
                our_species,
                archetype_distribution=archetype_dist.tolist(),
                variance=0.20,
            )
            opp_hydrated = hydrate_team(
                opp_species,
                archetype_distribution=archetype_dist.tolist(),
                variance=0.20,
            )

            assert len(our_hydrated) == 6
            assert len(opp_hydrated) == 6
            assert all(isinstance(mon, Pokemon) for mon in our_hydrated)
            assert all(isinstance(mon, Pokemon) for mon in opp_hydrated)

            our_strategies = enumerate_top_k_strategies(our_hydrated, k=3, opponent_team=opp_hydrated)
            opp_strategies = enumerate_top_k_strategies(opp_hydrated, k=3, opponent_team=our_hydrated)

            assert len(our_strategies) > 0, "should produce at least one strategy"
            assert len(opp_strategies) > 0

            final_mix, final_strategies = await nash_loop(
                our_hydrated,
                opp_hydrated,
                k_start=3,
                k_max=5,
                timeout_sec=10.0,
                battle_fn=_mock_battle,
            )

            assert len(final_strategies) > 0
            assert np.isclose(final_mix[:len(final_strategies)].sum(), 1.0, atol=0.05)

        asyncio.run(_run())

    def test_build_team_order_format(self) -> None:
        """Team order output is a valid /team command with species names."""
        species = ["snorlax", "charizard", "gyarados", "gengar", "alakazam", "dragonite"]
        strategy: Strategy = (0, 1, 2, 3, 4, 5)

        result = build_team_order(species, strategy)
        assert result.startswith("/team ")
        assert "|" in result
        parts = result.split(" ")
        assert len(parts) == 2
        assert len(parts[1].split("|")) == 6

    def test_shannon_entropy_known_values(self) -> None:
        """Shannon entropy produces correct values for known distributions."""
        uniform = np.array([0.25, 0.25, 0.25, 0.25])
        assert shannon_entropy(uniform) == pytest.approx(2.0, abs=0.05)

        peaked = np.array([1.0, 0.0, 0.0])
        assert shannon_entropy(peaked) == pytest.approx(0.0, abs=0.01)

    def test_pipeline_with_config_defaults(self) -> None:
        """SelectionConfig defaults are valid for pipeline operation."""
        config = SelectionConfig()
        assert config.procedural_variance == 0.20
        assert config.timeout_limit_sec == 60.0
        assert config.async_batch_size == 4
