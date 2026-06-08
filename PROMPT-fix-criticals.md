Fix all CRITICAL and HIGH issues found in the swarm audit of this repo. Work through them systematically.

## CRITICAL fixes (must do):

1. **src/llm_bias_testing/transformers.py:3** — `from keys import HF_token` is broken (keys.py doesn't exist). Replace with `os.environ.get("HF_TOKEN")`. Remove the keys.py dependency entirely.

2. **src/llm_bias_testing/ollama_setup.py:23-27** — After `subprocess.Popen(["ollama", "serve"])`, replace `time.sleep(2)` with a proper health check that polls `http://localhost:11434/api/tags` with retries and timeout.

3. **src/llm_bias_testing/ollama_setup.py:12-20** — Bare `except Exception: pass` in `_kill_existing_ollama` silently swallows all errors. Add logging.

4. **src/llm_bias_testing/ollama_setup.py:32-35** — Ollama server stdout/stderr sent to DEVNULL — crashes are invisible. Log stderr instead.

5. **pyproject.toml** — `pandas`, `matplotlib`, `tqdm` are used in main.py but NOT declared as dependencies. Add them.

## HIGH fixes (must do):

6. **main.py:115** — CSV written to disk on every loop iteration (3600 writes). Batch-collect records into a list, write once at end.

7. **main.py:112** — `pd.concat` in tight loop. Collect into list, concat once.

8. **main.py:58-61** — `record_exists` scans full DataFrame O(n) per check. Use a `set` of `(key, run)` tuples for O(1) lookups.

9. **src/llm_bias_testing/call_api.py** — `ollama.chat()` has no timeout. Add a timeout parameter.

10. **main.py:70-83** — No try/except around `model.predict()`. If model crashes mid-batch, entire process dies. Add error handling with graceful degradation (skip failed records, log error, continue).

11. **main.py:73** — `re.compile(r"(\d{1,3})/100")` called inside `process_cv_run` — recompiled 3600 times. Move to module level.

12. **compare_models.py:31-35** — `int(match.group(1))` where `match` can be `None`. Add None check.

13. **pyproject.toml** — `openai` is declared but never used. Remove it.

14. **pyproject.toml** — `ruff` is a dev-only tool but listed under `[project]dependencies`. Move to `[project.optional-dependencies]` dev group.

## Style fixes (do these too):

15. **src/llm_bias_testing/__init__.py:1-2** — Contains `def main() -> None: print("Hello from python-template!")`. Remove stale boilerplate.

16. Replace all `print()` calls in main.py with proper `logging` module usage.

## Rules:
- Keep all existing functionality working
- Do NOT change the core algorithm or data flow
- Do NOT add new dependencies beyond what's needed for fixes
- Add proper error handling throughout
- Use `logging` module instead of `print()`
- Write clean, readable code
- Run `uv run ruff check src tests` after all changes to verify no lint errors
