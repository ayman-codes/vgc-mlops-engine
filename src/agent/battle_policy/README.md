# Battle Policy Module

## Overview
This module implements a computationally optimized, heuristic-based decision engine for the VGC2 double battle environment. It utilizes independent slot Q-value mapping followed by an $O(K^2)$ joint synergistic resolution. 

## Component Architecture
* **`main.py`**: VGC2 framework entry point. Coordinates independent slot evaluation, executes heuristic pruning (Top-K selection), resolves the final joint matrix, and caches execution telemetry.
* **`heuristics/scoring.py`**: Computes independent, baseline Q-values for offensive, defensive, and switching actions utilizing standardized damage equivalence.
* **`heuristics/threat.py`**: Resolves dynamic turn-order mechanics (Trick Room, Tailwind, stat stages) and calculates absolute incoming lethal threat penalties based on speed tiers.
* **`heuristics/synergy.py`**: Evaluates cross-slot action modifiers (Focus Fire priority, Offensive/Defensive Support) to adjust independent baseline scores into a final joint Q-value.

## I/O & Telemetry
The `decision()` method returns a standard `List[BattleCommand]`. Execution telemetry (raw Q-values, command vectors, synergy contributions) is captured synchronously and exposed via the `get_telemetry()` stateful getter for downstream MLOps pipeline ingestion.

## Execution Pipeline
1. Bind engine parameters via native object initialization.
2. Extract active slot indices and valid targets.
3. Generate independent candidate actions and baseline Q-values for Slot 0 and Slot 1.
4. Truncate candidate arrays to the Top-K actions based on baseline scores.
5. Evaluate the $K \times K$ combination matrix for synergistic modifiers.
6. Return the highest-scoring command tuple and write to the telemetry buffer.