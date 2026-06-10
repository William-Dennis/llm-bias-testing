# Contributing to SLM Bias Testing

## Development Setup

```bash
git clone git@github.com:William-Dennis/slm-bias-testing.git
cd slm-bias-testing
uv sync --extra dev
```

## Workflow

This repo follows a **feature-branch → PR → merge** model. Never commit directly to `main`.

1. Create a branch from `main`:
   ```bash
   git checkout main && git pull
   git checkout -b feat/your-feature
   ```

2. Make changes, write tests first (TDD), then implement.

3. Run checks locally before pushing:
   ```bash
   uv run ruff check src tests
   uv run ruff format --check src tests
   uv run mypy src/slm_bias_testing
   uv run pytest
   ```

4. Push and open a PR targeting `main`.

5. All CI checks must pass before merge. Address review feedback promptly.

## Code Quality Standards

- **Linting:** Ruff (pycodestyle, pyflakes, isort, bugbear, simplify)
- **Formatting:** Ruff formatter
- **Type checking:** Mypy with `disallow_untyped_defs`
- **Testing:** Pytest with coverage for all new code
- **SOLID:** Single responsibility per module/class. No mutable global state.

## Commit Convention

```
type: concise subject line

Optional body.
```

Types: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`

## Running Benchmarks

Benchmarks require Ollama running locally:

```bash
ollama serve &
uv run python -m slm_bias_testing.runner gemma3-1b --benchmark cv-screening
```

Or run the full overnight suite:

```bash
bash scripts/overnight_run.sh
```
