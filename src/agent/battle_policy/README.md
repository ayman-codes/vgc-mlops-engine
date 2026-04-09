# Battle Policy Module

## Overview
This module implements a computationally optimized decision engine for the VGC2 double battle environment. It executes independent slot Q-value mapping followed by exhaustive joint synergistic resolution. This architecture establishes the deterministic baseline required for downstream Deep Reinforcement Learning (DRL) pre-training and self-play curriculum bootstrapping.

## Component Architecture
* **`main.py`**: Orchestration entry point. Coordinates independent slot evaluation, resolves the complete joint action matrix to preserve continuous Q-value gradients, and caches execution telemetry.
* **`heuristics/scoring.py`**: Computes independent, baseline Q-values for offensive, defensive, and switching actions. Utilizes exact, dynamic HP-equivalent damage proxies instead of arbitrary static multipliers.
* **`heuristics/threat.py`**: Resolves dynamic turn-order mechanics (Trick Room, Tailwind, stat stages). Calculates incoming lethal threat penalties scaled by precise type-effectiveness modifiers.
* **`heuristics/synergy.py`**: Evaluates cross-slot action modifiers (focus-fire priority, protective support) to adjust independent baseline scores into final joint Q-values.
* **`utils/type_chart.py`**: Supplies absolute VGC2 type-effectiveness multipliers (weaknesses, resistances, immunities) for threat and damage resolution.
* **`policies/`**: Contains the DRL curriculum progression variants (Greedy, Epsilon-Greedy, Softmax) utilized for network exploration and validation.

## I/O & Telemetry
Enforces strict interface parity with the VGC2 environment. Ingests state views and returns a validated `List[BattleCommand]`. Execution telemetry (raw Q-values, command vectors, synergy contributions, and softmax probability distributions) is captured synchronously and exposed via the `get_telemetry()` stateful getter for MLOps pipeline ingestion.

## Execution Pipeline
1. Bind engine parameters via native object initialization and apply type-chart matrices.
2. Extract active slot indices and valid targeting vectors.
3. Generate independent candidate actions and compute baseline Q-values for Slot 0 and Slot 1.
4. Evaluate the complete combinatorial matrix for synergistic modifiers. Arbitrary Top-K truncation is strictly bypassed to prevent pruning of joint-utility actions.
5. Route the combined Q-value matrix through the active curriculum policy to extract the optimal or exploratory action.
6. Return the standardized command tuple and flush data to the telemetry buffer.