# Selection Policy Module

## Overview
This module executes the Selection Policy for the VGC2 framework. It implements a game-theoretic evaluation pipeline, serving as the production foundation for Bayesian Hidden State Inference and Nash Equilibrium solvers. It replaces exhaustive heuristic simulations with statistically grounded probabilistic modeling.

## Component Architecture

* **`main.py`**: Orchestration entry point. Ingests multidimensional telemetry and yields optimized action arrays.
* **`inference/`**: 
    * `bayesian.py`: Computes posterior probabilities of unrevealed opponent entities utilizing teammate co-occurrence prior matrices.
    * `gmm.py`: Deploys continuous feature space vectorization to map historical statistics to latent meta-game clusters.
    * `payoff.py`: Generates zero-sum payoff matrices mapping allied permutations against opponent probability distributions.
    * `nash.py`: Solves payoff matrices utilizing SciPy linear programming (`linprog`) to extract optimal mixed strategy distributions.
* **`heuristics/`**: 
    * `archetype.py`: Extrapolates hidden parameters and translates base stats into engine-compliant `Pokemon` builds via threshold meta-heuristics.
    * `scoring.py`: Computes absolute deterministic utility scalars for non-damaging operations, including status, weather, and terrain control.
    * `matchup.py`: Executes decoupled $O(N^2)$ simulation sub-tournaments for baseline state value estimation.

## I/O & Telemetry
Enforces strict interface parity with `vgc2.agent.SelectionPolicy`. Ingests player `Team` and opponent `TeamView` structures. Outputs a validated `SelectionCommand` specifying the optimized active roster indices.

## Execution Pipeline
1. **State Ingestion:** Parse $6 \times 6$ `TeamPreview` environment matrices.
2. **Hidden State Estimation:** Execute Bayesian recursive updates to impute unrevealed opponent configurations.
3. **Archetype Projection:** Instantiate engine-compliant opponent models via GMM cluster centroids and `predict_opponent_builds` logic.
4. **Simulation Matrix Construction:** Generate zero-sum payoff matrices via combinatorial execution of allied pairs against predicted opponent pairs utilizing a baseline `GreedyBattlePolicy`.
5. **Equilibrium Resolution:** Compute the optimal selection strategy via Linear Programming and output the `SelectionCommand`.