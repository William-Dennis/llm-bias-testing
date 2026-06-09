# LLM Bias Testing

Evaluate bias in small language models across multiple benchmarks.

## Quick Start

```bash
# Install
uv sync --extra dev

# Run all benchmarks on one model
uv run python -m llm_bias_testing.runner smollm2-135m --benchmark all

# Run specific benchmark with limited samples
uv run python -m llm_bias_testing.runner smollm2-135m --benchmark stereoset --max-samples 20

# Batch run experiment script
uv run python scripts/run_experiments.py --models smollm2-135m,smollm2-360m --benchmarks stereoset --max-samples 20
```

## Prerequisites

- [Ollama](https://ollama.ai) — all models run locally via Ollama
- `uv` (or `pip`) for Python dependencies

## Benchmarks

| Benchmark | Status | Description |
|---|---|---|
| `cv-screening` | ✅ Working | Detects bias in CV scoring by name/university/A-levels |
| `stereoset` | ✅ Working | Standardised stereotype score by category (gender, race, religion, profession) |
| `demographic-bias` | ✅ Working | Measures output length differences across 8 demographic groups (400 prompts) |
| `crows-pairs` | ❌ Unavailable | HuggingFace dataset deprecated — not accessible |
| `bbq` | ❌ Unavailable | HuggingFace dataset not found |

## CLI Reference

### `runner.py` (recommended)

```bash
uv run python -m llm_bias_testing.runner <model_name> [options]

Arguments:
  model_name               Registered model name (e.g. smollm2-135m, qwen25-05b)

Options:
  --benchmark {cv-screening,stereoset,demographic-bias,all}
  --output-dir DIR         Results directory (default: results/)
  --timeout SECS           Per-model timeout (default: 1800)
  --max-samples N          Limit samples for testing (default: all)
```

### `main.py` (legacy, CV-only)

```bash
uv run python main.py --model smollm2-135m
```

### `scripts/run_experiments.py` (batch runner)

```bash
uv run python scripts/run_experiments.py \
  --models smollm2-135m,smollm2-360m \
  --benchmarks stereoset,demographic-bias \
  --max-samples 20 --timeout 300
```

## Models

11 registered models in `src/llm_bias_testing/registry.py`. Models under 1B params:

| Name | Ollama Tag | Params | Release | Family |
|---|---|---|---|---|
| smollm-135m | smollm:135m | 135M | 2024-07 | huggingface |
| smollm-360m | smollm:360m | 360M | 2024-07 | huggingface |
| smollm2-135m | smollm2:135m | 135M | 2024-11 | huggingface |
| smollm2-360m | smollm2:360m | 360M | 2024-11 | huggingface |
| qwen25-05b | qwen2.5:0.5b | 500M | 2024-09 | alibaba |
| qwen35-08b | qwen3.5:0.8b | 800M | 2025-05 | alibaba |

## Experiment Results (June 2026)

### StereoSet (6 models, 20 samples each)

| Model | StereoScore |
|---|---|
| smollm2-135m | 5.0% |
| smollm-135m | 20.0% |
| qwen2.5:1.5b | 20.0% |
| smollm-360m | 25.0% |
| smollm2-360m | 35.0% |
| qwen2.5:0.5b | 35.0% |

### Demographic Bias (smollm2-135m, 400 prompts)

| Group | Avg Output Length |
|---|---|
| religion_muslim | 658.6 |
| age_old | 637.6 |
| age_young | 603.7 |
| gender_male | 569.4 |
| gender_female | 561.4 |
| race_white | 513.0 |
| race_black | 507.3 |
| religion_christian | 462.9 |

Key finding: Muslim prompts get 42% more output than Christian prompts.
Gender and race differences are small.

## Tests

```bash
uv run pytest tests/ -q
uv run ruff check src tests
```

## Project Structure

```
src/llm_bias_testing/
  registry.py       — Model definitions (name → ollama tag)
  runner.py         — CLI entry point for running benchmarks
  benchmark.py      — CV screening benchmark (legacy)
  call_api.py       — Ollama model API client
  ollama_setup.py   — Ollama server management
  benchmarks/
    __init__.py      — BaseBenchmark abstract class
    stereoset.py     — StereoSet benchmark
    demographic_bias.py — Demographic bias benchmark
    crows_pairs.py   — Unavailable (deprecated dataset)
    bbq.py           — Unavailable (dataset not found)
```

## Issues

- Open issues #1 and #3 (Add OpenAI/Gemini API) are superseded — the repo now uses Ollama for all models.
- CrowS-Pairs and BBQ benchmarks are non-functional due to HuggingFace dataset deprecation.
