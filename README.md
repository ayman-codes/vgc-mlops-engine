<p align="center">
  <h1 align="center">VGC-MLOps Engine</h1>
  <p align="center"><strong>Enterprise-grade Multi-Agent Reinforcement Learning for Competitive Pokémon VGC</strong></p>
  <p align="center">
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.14-blue?logo=python&logoColor=white" alt="Python 3.14"></a>
    <a href=".github/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/your-repo/vgc-mlops-engine/ci.yml?branch=main&logo=github" alt="CI"></a>
    <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/badge/code%20style-ruff-000000?logo=ruff" alt="Ruff"></a>
    <a href="https://mypy-lang.org/"><img src="https://img.shields.io/badge/typed-mypy-039dfc" alt="mypy"></a>
    <a href="https://mlflow.org/"><img src="https://img.shields.io/badge/MLflow-tracked-0194E2?logo=mlflow" alt="MLflow"></a>
    <a href="https://prometheus.io/"><img src="https://img.shields.io/badge/monitoring-Prometheus%2FGrafana-E6522C?logo=prometheus" alt="Monitoring"></a>
    <a href="https://www.docker.com/"><img src="https://img.shields.io/badge/infra-Docker-2496ED?logo=docker" alt="Docker"></a>
    <br>
    <a href="https://poke-env.readthedocs.io/"><img src="https://img.shields.io/badge/sim-poke--env-red" alt="poke-env"></a>
    <a href="https://pypi.org/project/vgc-bench/"><img src="https://img.shields.io/badge/bench-vgc--bench-purple" alt="vgc-bench"></a>
    <a href="https://scipy.org/"><img src="https://img.shields.io/badge/Nash-SciPy%20LP-8CAE00?logo=scipy" alt="Nash"></a>
  </p>
</p>

---

## Overview

The VGC-MLOps Engine models Pokémon VGC as a **multi-agent, partially observable stochastic MDP** and implements a production-grade AI pipeline spanning **game theory, deep reinforcement learning, Bayesian inference, evolutionary algorithms, and enterprise MLOps**.

Simulation runs on local Pokémon Showdown servers via `poke-env` with `vgc-bench` benchmarking — zero-latency, deterministic, fully parallelizable.

---

## Core Competencies

| Domain | Skills |
|--------|--------|
| **Algorithms** | Evolutionary Algorithms, Bayesian Inference & Naive Bayes, Gaussian Mixture Models (GMM), Nash Equilibrium (Linear Programming), Deep Q-Networks (DQN), Proximal Policy Optimization (PPO), Behavior Cloning (Imitation Learning), Policy Space Response Oracles (PSRO), Self-Play, Fictitious Play, Double Oracle, Epsilon-Greedy / Softmax Exploration, Mixed-Strategy Game Theory |
| **Services & Architecture** | MLflow (Experiment Tracking, Model Registry), Prometheus + Grafana (Observability, Metrics, Dashboards), n8n (Workflow Orchestration, Event-Driven Pipelines), REST APIs (External Integrations), Cloudflare Tunnels (Secure Remote Access), Container Orchestration (Docker Compose, Multi-Stage Builds) |
| **Data Engineering** | Bronze → Silver → Gold Medallion Architecture, DuckDB (Out-of-Core Parquet Serialization), HuggingFace Hub (Dataset Versioning), Fuzzy Entity Resolution (`thefuzz`), Cross-Schema Hydration, One-Hot Encoding, Float32 Tensor Generation, Data Quality Gates (Deficit Threshold Enforcement) |
| **Machine Learning & AI** | scikit-learn (GMM, Clustering, Silhouette Score), SciPy (Linear Programming, `linprog` HiGHS), Population-Based Training, Experience Replay, Frame Stacking, Metagame Archetype Discovery, Procedural Variance Injection |
| **Reinforcement Learning** | MARL (Multi-Agent RL), PPO, DQN, PSRO League Training, Self-Play, Fictitious Play, Policy Exploitation, Boltzmann Exploration |
| **MLOps & DevOps** | CI/CD (GitHub Actions: Ruff, mypy, pytest + Hypothesis), Docker (Multi-Stage, Slim Images), `uv` (Fast Python Package Manager, Frozen Lockfiles), Prometheus Metrics Export, Grafana Dashboards, MLflow Tracking & Registry, A/B Testing (Automated Model Comparison), Optuna (Hyperparameter Tuning) |
| **Software Engineering** | mypy (Strict Mode), Ruff (Linting), pytest + Hypothesis (Property-Based Testing), Pydantic (Configuration Validation), Modular Architecture (Policy Engines, ETL Layers, Plugins), Type Hinting (Strict `mypy` Compliance) |
| **Infrastructure** | Docker Compose (Multi-Service Orchestration), Cloudflare Tunnels (Zero-Trust Networking), Containerized Telemetry Stack, Cross-Platform Deployment (Windows/WSL2/Linux) |
| **Domain Expertise** | Competitive Pokémon VGC (Video Game Championships), Game Theory (Zero-Sum Games, Nash Equilibrium), Turn-Based Strategy AI, Hidden Information Inference, Metagame Analysis |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        VGC-MLOps ENGINE                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────┐   ┌──────────────────┐   ┌─────────────────────┐  │
│  │   BATTLE POLICY  │   │  SELECTION POLICY│   │  TEAMBUILD POLICY   │  │
│  │  (Combat Engine)  │   │ (Team Preview)   │   │  (Draft Engine)     │  │
│  ├──────────────────┤   ├──────────────────┤   ├─────────────────────┤  │
│  │ • Joint-Action Q │   │ • Bayesian Inf.  │   │ • Genetic Algorithm │  │
│  │ • Synergy Matrix │   │ • Payoff Matrix  │   │ • Match-up Predictor│  │
│  │ • DQN / PPO      │   │ • Nash Equilib.  │   │ • Adversarial Eval  │  │
│  │ • PSRO / Self-Play│  │ • GMM Clustering │   │                     │  │
│  │ • BC Imitation   │   │ • Variance Inj.  │   │                     │  │
│  └────────┬─────────┘   └────────┬─────────┘   └─────────────────────┘  │
│           │                      │                                       │
│  ┌────────▼──────────────────────▼──────────────────────────────────────┐│
│  │                     DATA ENGINEERING (Bronze→Silver→Gold)            ││
│  │  DuckDB  │  HuggingFace  │  PokeAPI  │  Smogon  │  FuzzyMatch       ││
│  └──────────────────────────────────────────────────────────────────────┘│
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐│
│  │              MLOps & OBSERVABILITY  LAYER                            ││
│  │  MLflow  │  Prometheus  │  Grafana  │  n8n  │  Docker  │  GitHub CI ││
│  └──────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
```

### Policy Breakdown

**Battle Policy** — Turn-by-turn double-battle decisions. Each slot is evaluated independently for all actions (attack/protect/switch), then a cross-slot joint synergy matrix resolves optimal action pairs. Baseline exploration strategies (Epsilon-Greedy, Softmax) feed the PSRO league for emergent strategy discovery.

**Selection Policy** — During Team Preview, a Bayesian engine hydrates hidden opponent states from co-occurrence priors. A zero-sum payoff matrix is constructed via parallel sub-tournaments, then resolved to a Nash equilibrium mixed strategy — returning the optimal 4-Pokémon active roster.

**Teambuild Policy** — Evolutionary/genetic algorithms evolve full 6-Pokémon rosters against metagame simulations. Deep match-up predictors serve as fitness evaluators without expensive battle simulations.

---

## Quick Start

```bash
# Clone & install
git clone https://github.com/your-org/vgc-mlops-engine
cd vgc-mlops-engine
uv sync --frozen

# Run benchmarks
python -m benchmarks.evaluate_battle
python -m benchmarks.evaluate_selection

# Launch full MLOps stack
docker compose up --build
```

---

## Project Phases

| Phase | Status | Deliverables |
|-------|--------|-------------|
| **1 — MLOps Infrastructure** | ✅ Complete | Docker stack (MLflow, Prometheus, Grafana), CI/CD, `ruff`/`mypy`/`pytest` |
| **2 — Selection Policy** | 🔄 In Progress | Bayesian priors, payoff matrices, Nash LP, GMM archetype analysis |
| **3 — Battle Policy** | 📋 Planned | BC initializer, DQN + PPO, Self-Play, PSRO league |
| **4 — Teambuild Policy** | 📋 Planned | Evolutionary draft, adversarial balance agents |

---

## Repository Structure

```
src/
├── agent/
│   ├── battle_policy/        # Joint-action heuristic + DRL baselines
│   │   ├── heuristics/       # Scoring, threat, synergy, type chart
│   │   ├── baselines/        # Epsilon-Greedy, Softmax
│   │   └── utils/            # Type effectiveness matrix
│   ├── selection_policy/     # Game theory + Bayesian inference
│   │   ├── inference/        # Bayesian, GMM, Nash, Payoff
│   │   └── heuristics/       # Archetype, matchup, scoring
│   ├── core_player.py        # Abstract VGC agent
│   └── teambuilder.py        # Evocation draft stub
├── data_ingestion/           # Bronze layer: API extraction
├── data_processing/          # Silver→Gold: hydration, tensors
├── config/                   # YAML weights, Pydantic schemas
benchmarks/                   # Cross-eval tournaments + MLflow logging
tests/                        # pytest + Hypothesis property tests
infrastructure/               # Docker, n8n, Cloudflare
```

---

## Benchmarks

| Benchmark | Description | Metrics Tracked |
|-----------|-------------|----------------|
| **Battle** | 6-policy cross-eval tournament | Win rates vs Random, MaxBasePower, SimpleHeuristics |
| **Selection** | Bayesian Nash vs baselines | Matrix resolution time, win rate delta |
| **Bayesian** | Offline prediction accuracy | Item/ability prediction accuracy |
| **GMM** | Metagame clustering quality | Silhouette score, log-likelihood |

All benchmark results are logged to MLflow and visualized in Grafana dashboards.

---

## License

Custom [VGC-AI Tournament Exclusion License](LICENSE) — personal and educational use permitted; commercial and tournament use prohibited.

---

<p align="center">
  <sub>Built with Python 3.14 • poke-env • vgc-bench • MLflow • Docker • SciPy</sub>
</p>
