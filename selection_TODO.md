# Selection Policy — Issues & Implementation Notes

## Issues Encountered

### double_oracle.py

1. **Recursive import** — `_showdown_battle` tried to import `_pokemon_to_showdown` from its own module. Fixed by defining `_pokemon_to_showdown` before `_showdown_battle`.

2. **Move data lookup** — Initially used `base.get(move.name, {}).get("basePower", 80)` which looks up moves in `baseStats`, not the moves database. Fixed by adding `_get_move_data` that queries `GenData.moves`.

3. **Unused variable `dex`** — Ruff F841 in `_pokemon_to_showdown`. Removed.

4. **Type annotation errors** — `_GEN_DATA_CACHE` typed as `dict` but holds `GenData` object. Changed to `Any`. Multiple `Returning Any` errors from poke_env dict access (`pokedex.get`, `moves.get`). Fixed with `str` intermediate variables and `Any` return types.

5. **Missing `stop_listening`** — `Player` has no `stop_listening` method (it's on `PSClient`). Removed cleanup `finally` block.

6. **Wrong import for AccountConfiguration** — Used `PlayerConfiguration` from `poke_env.player` but it should be `AccountConfiguration` from `poke_env`.

### optimizer.py

7. **Undefined variable `opp_team`** — Typo in `nash_loop`: used `opp_team` instead of `opponent_team`. Fixed.

8. **Unused `run` variable** — MLflow context manager `as run:` assigned unused variable. Removed `as run`.

9. **Unused import `find_best_response`** — Removed after `_expand_set` was rewritten to `_next_strategy`.

10. **`_expand_set` logic wrong** — Used a diagonal mock payoff matrix and incorrect index comparison. Replaced with `_next_strategy` that enumerates fresh strategies not in the current set.

11. **`Returning Any` from scipy linprog** — `result.x` is untyped by scipy stubs. Fixed with explicit `np.asarray(..., dtype=np.float64)` and typed intermediate variable.

### general

12. **mypy `no-any-return` on poke_env dict access** — Any return from `.get()` requires intermediate typed variable or explicit cast.

13. **Showdown server format unknown** — Using `gen9randombattle` as fallback because the exact VGC format name for the local server is unknown. May fail if the server doesn't support custom teams for random battle.

14. **`_showdown_battle` unused inside `double_oracle.py`** — The function is only used as default in `expand_matrix_async`. May need to add `# type: ignore` for the import inside the function (lazy import of poke_env).

## Remaining Work

- [ ] Create `main.py` (SelectionPolicyPlayer with teampreview hook)
- [ ] Phase 4.2: Shannon entropy telemetry + GMM model registration
- [ ] Tests: `test_double_oracle.py`, `test_optimizer.py`, `test_main.py`
- [ ] Quality gate: ruff → mypy → pytest
