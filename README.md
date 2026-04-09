# VGC-MLOps Engine: Enterprise AI for Competitive Pokémon

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![CI/CD: GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-success.svg)](.github/workflows)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Transforming heuristic baselines into a production-grade Multi-Agent Reinforcement Learning (MARL) pipeline.**

## Overview

The VGC-MLOps Engine is an advanced computational architecture designed to model and simulate the Pokémon Video Game Championships (VGC). It functions as a complete multi-agent framework capable of executing complex strategic decision-making in a highly stochastic, partially observable Markov Decision Process (MDP) environment.

This project implements a rigorous enterprise data pipeline, robust machine learning models (GMMs, Bayesian Inference, PPO), event-driven orchestration, and strict MLOps engineering standards.

## Technical Architecture

The framework is decoupled into core policy engines and supporting infrastructure:

1.  **Orchestration Layer (n8n):** Functions as the control plane. Executes cron-scheduled scraping, event-driven data ingestion, AWS S3 routing, pipeline incident alerting, and workflow automation.
2.  **Battle Policy (Combat Engine):** The curriculum establishes a progressive baseline (Greedy $\rightarrow$ Epsilon-Greedy $\rightarrow$ Softmax $\rightarrow$ Heuristic) to pre-train Deep Reinforcement Learning (DRL) networks trained on all of the other constructed battle policies.
3.  **Selection Policy (Inference Engine):** Replaces static predictions with statistically grounded mathematical models. Utilizes Gaussian Mixture Models (GMM) for unsupervised archetype discovery and a Bayesian engine for hidden state estimation. Output distributions resolve via a SciPy linear programming Nash Equilibrium solver.
4.  **Teambuild Policy (Optimization Engine):** Implements the Heuristic-Evolutionary-Simulation Funnel (HESF). Navigates combinatorial synergy spaces using mass-pruning linear optimization and a Genetic Algorithm (GA) seeded by GMM cluster centroids.
5.  **Quality Assurance:** Enforces strict pipeline integrity via `ruff` for high-performance linting and `hypothesis` for property-based testing of mathematical bounds and array shapes.

## Highlighted Skills & Competencies

This repository demonstrates capabilities required for Senior Data Engineering and MLOps/Cloud Architect roles.

* **Machine Learning & Statistics:** Multi-Agent Reinforcement Learning (MARL), Gaussian Mixture Models (GMM), Naive Bayes Inference, Game Theory (Nash Equilibrium, Minimax Regret), Genetic Algorithms (NSGA-II).
* **Data Engineering (ETL Pipelines):** Event-driven orchestration (n8n), JSON schema unnesting, regex parsing of protocol streams, dimensional modeling, localized caching strategies (SQLite/DuckDB), and API integration (LimitlessTCG, PokeAPI).
* **Software Engineering:** Algorithm optimization, Object-Oriented design, parallelized batch processing (`concurrent.futures`), property-based testing (`hypothesis`), static typing (`mypy`), and strict linting (`ruff`).
* **MLOps & Infrastructure:** Dependency management (`pyproject.toml`, `uv`), Multi-stage Docker containerization, GitHub Actions CI/CD pipelines, and experiment tracking (MLflow).

## Repository Structure

```text
vgc-mlops-engine/
├── workflows/                  # n8n orchestration JSON templates
├── src/
│   ├── agent/
│   │   ├── battle_policy/      # Action execution and DRL curriculum
│   │   ├── selection_policy/   # GMM, Bayesian inference, and Nash solvers
│   │   └── teambuild_policy/   # HESF pipeline and Genetic Algorithms
│   ├── etl/                    # Extraction, Transformation, and Loading pipelines
│   └── config/                 # YAML-based decoupled configurations
├── tests/                      # Pytest and Hypothesis property-based test suites
├── pyproject.toml              # UV/Poetry dependency definitions
└── Dockerfile                  # Multi-stage build definitions
```

## Execution Queue (TODO)

### 1. Data Orchestration & ETL (n8n & Pipeline)
- [ ] Deploy n8n workflows for cron-scheduled LimitlessTCG `/tournaments` metadata extraction and AWS S3 landing zone routing.
- [ ] Construct the protocol parser for pipe-delimited simulator streams to map State-Action-Reward tuples.
- [ ] Transform Smogon Chaos JSON `Spreads` objects into normalized $1 \times 6$ numerical feature vectors.
- [ ] Populate local SQLite/DuckDB relational tables utilizing raw PokeAPI dumps to establish an offline continuous data flow.

### 2. Selection Policy Refinement
- [ ] Ingest the transformed feature vectors from the ETL phase into `ArchetypeGMM.fit()` to lock statistical cluster parameters.
- [ ] Calculate the normalized $N \times N$ adjacency matrix from the Smogon `Teammates` dictionary to instantiate the `BayesianHiddenStatePredictor` prior matrix.
- [ ] Expand the hypothesis test suite to validate payoff matrix dimensional stability under edge-case team sizes.

### 3. Teambuilding Implementation
- [ ] Implement Stage 1: Representative Meta-Pruning via Potential Damage Output (PDO) mapping.
- [ ] Implement Stage 2: Genetic Algorithm infrastructure (Tournament Selection, Crossover, Probability-Weighted Mutation).
- [ ] Configure `concurrent.futures` batching to distribute simulation workloads.

### 4. Continuous Integration
- [ ] Enforce `ruff` static analysis checks as terminal failure conditions in GitHub Actions.
- [ ] Integrate MLflow logging for automated model hyperparameter tracking.
