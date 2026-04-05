# Selection Policy Module

## Overview
This module executes the Selection Policy for the VGC2 framework. It creates a modular, heuristically-driven evaluation pipeline. It serves as the architectural foundation for subsequent Bayesian Hidden State Inference and Nash Equilibrium solvers. The first version was an extensive simulation tournament ran on the elite (12) to determine the best 6.

## Component Architecture
* **`main.py`**: VGC2 entry point. Orchestrates opponent build prediction, team combination generation, and sub-tournament resolution to select the optimal active roster.
* **`heuristics/scoring.py`**: Computes deterministic utility scores for non-damaging moves, field effects, and status conditions via normalized damage equivalence.
* **`heuristics/archetype.py`**: Predicts opponent move-sets and constructs statistical counter-build archetypes utilizing physical/special stat thresholds and VGC meta-heuristics.
* **`heuristics/matchup.py`**: Executes decoupled simulation sub-tournaments, comparing allied permutations against predicted opponent archetype matrices.

## I/O & Telemetry
Maintains strict interface parity with `vgc2.agent.SelectionPolicy`. Ingests player and opponent `Team` / `TeamView` objects. Returns a `SelectionCommand` specifying the optimized active Pokémon indices.

## Execution Pipeline
1. Ingest opponent team preview objects.
2. Execute `predict_opponent_builds` to extrapolate hidden stats, natures, and unrevealed moves.
3. Generate combinatorial matrices mapping allied team pairs against predicted opponent pairs.
4. Run `run_sub_tournament` utilizing `GreedyBattlePolicy` to compute baseline win probabilities.
5. Rank permutations and yield the final `SelectionCommand` array.