# VGC-MLOps Engine: Enterprise AI for Competitive Pokémon

[![Python 3.14](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![CI/CD: GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-success.svg)](.github/workflows)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Transforming heuristic baselines into a production-grade Multi-Agent Reinforcement Learning (MARL) pipeline.**

## Overview

The VGC-MLOps Engine is an advanced computational architecture designed to model and simulate the Pokémon Video Game Championships (VGC). It functions as a complete multi-agent framework capable of executing complex strategic decision-making in a highly stochastic, partially observable Markov Decision Process (MDP) environment.

Our simulation and benchmarking architecture relies on the `poke-env` programmatic interface and the `vgc-bench` multi-agent evaluation framework. This infrastructure allows our AI models to generalize across highly diverse meta-games by competing directly on locally hosted, zero-latency NodeJS Pokémon Showdown servers.

## Technical Architecture & The Algorithmic Arsenal

The framework is decoupled into core policy engines and enterprise MLOps infrastructure. This project is built as a premier demonstration of deploying complex algorithms in a production setting.

### 1. Orchestration & MLOps Layer
Functions as the control plane for the entire engine.
*   **MLflow:** For robust hyperparameter and telemetry tracking (e.g., matrix regret, matrix resolution time, inference confidence).
*   **Prometheus & Grafana:** For real-time hardware monitoring, system latency, and model performance metrics.
*   **n8n:** Containerized workflow orchestration for event-driven, offline data ingestion.

### 2. Battle Policy (Combat & Tactics Engine)
Executes optimal turn-by-turn combat decisions in the partially observable `poke-env` environment.
*   **Behavior Cloning (BC):** Imitation Learning model pre-trained on human `vgc-battle-logs` to initialize the Reinforcement Learning agent with competent, non-random weights.
*   **Deep Q-Networks (DQN) & Experience Replay:** RL architecture utilizing Bellman equations and target networks, coupled with an Experience Replay Buffer to break correlation between consecutive battle samples.
*   **Proximal Policy Optimization (PPO) & Self-Play (SP):** The core Reinforcement Learning architectures utilized to calculate optimal expected future rewards.
*   **Frame Stacking:** Data-transformation technique stacking consecutive state frames, allowing neural networks to infer temporal sequences and momentum rather than static snapshots.
*   **Policy Space Response Oracles (PSRO) & Policy Exploitation:** Population-based multi-agent meta-algorithms. Orchestrates the macro-training loop by probabilistically sampling opponents from a dynamic pool of Fictitious Play, Double Oracle, and historical baseline checkpoints to ensure global strategy generalization and prevent catastrophic forgetting.

### 3. Selection Policy (Team Preview Engine)
Resolves the combinatorial 4-out-of-6 selection matrix problem.
*   **Bayesian Inference Engine:** Calculates posterior probabilities of hidden opponent states (masked EVs, items, abilities) during Team Preview utilizing pre-compiled offline co-occurrence adjacency matrices.
*   **Zero-Sum Payoff Matrices:** Mathematical construct mapping terminal damage outputs of all 15x15 possible Team Preview lead combinations, calculated via parallel batch simulations.
*   **Nash Equilibrium Resolution:** Applies SciPy linear programming solvers to the Payoff Matrix to extract optimal, unexploitable mixed-strategy deployment vectors.
*   **Procedural Variance Injection (20% Rule):** Algorithmic mutation (stat inversion, move substitution) applied to baseline opponent arrays. Mathematically forces Game-Theoretic solvers to account for chaotic, off-meta configurations, preventing brittle matrix generation.
*   **Gaussian Mixture Models (GMM):** Utilized exclusively for offline Exploratory Data Analysis (EDA) to cluster the metagame into distinct archetypes (e.g., Trick Room, Hyper-Offense) for data visualization.

### 4. Teambuild Policy (Optimization Engine)
Operates on a Meta-Game Simulation track to evaluate and draft full 6-Pokémon rosters.
*   **Genetic / Evolutionary Algorithms (GA):** Treats rosters as chromosomes, iteratively evolving Generation 0 seeds by selecting combinations with high statistical win-rates.
*   **Deep Match-up Predictors:** Neural network fitness evaluators that predict team-vs-team win rates purely from structural composition without executing computationally expensive battle simulations.

## Master Execution Plan

### Phase 1: MLOps Infrastructure & Architecture Verification (Completed)
- [x] Configure localized `poke-env` environments with `vgc-bench` benchmarking baselines.
- [x] Establish containerized telemetry infrastructure (MLflow, Prometheus, Grafana).
- [x] Ensure CI/CD compliance (`ruff`, `mypy`, `pytest` + Hypothesis).

### Phase 2: Selection Policy (Game Theory & Inference)
- [ ] Parse `vgc-battle-logs` to construct conditional probability adjacency matrices (Items/Abilities/EVs).
- [ ] Implement the `BayesianInferenceEngine` to hydrate opponent Team Preview data dynamically with Procedural Variance Injection.
- [ ] Integrate parallel local battles to calculate zero-sum payoff matrices for 15x15 lead combinations.
- [ ] Implement SciPy linear programming solvers to extract optimal mixed-strategy deployment vectors.

### Phase 3: Battle Policy (MARL & Combat Mechanics)
- [ ] Train Behavior Cloning (BC) initializers on human `vgc-battle-logs` with Frame Stacking.
- [ ] Implement DQN and PPO algorithms tailored to the `poke-env` action space.
- [ ] Establish the Competitor League environment and execute PSRO training loops.

### Phase 4: Teambuild Policy (Evolutionary Draft Engine)
- [ ] Integrate offline relational ETL pipelines to output deterministic Generation 0 seeds.
- [ ] Build the Evolutionary Algorithm loop (Tournament Selection, Crossover, Probability-Weighted Mutation).
- [ ] Deploy Adversarial Balance Agents to evaluate roster meta-entropy and prevent strategy stagnation.
