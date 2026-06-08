import argparse
import json
import logging
import os
import subprocess
import sys

from llm_bias_testing.registry import MODELS, get_model

logger = logging.getLogger(__name__)


def pull_model(ollama_tag: str) -> bool:
    """Pull an Ollama model image if not already present."""
    logger.info("Pulling model %s ...", ollama_tag)
    result = subprocess.run(
        ["ollama", "pull", ollama_tag],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("Failed to pull model %s: %s", ollama_tag, result.stderr)
        return False
    logger.info("Successfully pulled %s", ollama_tag)
    return True


def run_benchmark_for_model(
    model_name: str,
    benchmark: str,
    base_output_dir: str,
    timeout: int,
) -> None:
    """Run the benchmark for a single model with resume support."""
    model_config = get_model(model_name)
    ollama_tag = model_config["ollama_tag"]

    results_dir = os.path.join(base_output_dir, model_name, benchmark)
    results_file = os.path.join(results_dir, "results.json")

    if os.path.exists(results_file):
        logger.info("Results already exist for %s/%s — skipping", model_name, benchmark)
        return

    if not pull_model(ollama_tag):
        logger.error("Skipping %s due to pull failure", model_name)
        return

    os.makedirs(results_dir, exist_ok=True)

    logger.info("Running benchmark %s with model %s ...", benchmark, model_name)

    from main import run_benchmark

    df = run_benchmark(
        model_name=ollama_tag,
        output_dir=results_dir,
        timeout=timeout,
    )

    summary = {
        "model": model_name,
        "ollama_tag": ollama_tag,
        "benchmark": benchmark,
        "n_records": len(df) if df is not None else 0,
        "mean_score": float(df["score"].mean()) if df is not None and not df.empty else None,
        "std_score": float(df["score"].std()) if df is not None and not df.empty else None,
    }
    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved results for %s/%s (%d records)", model_name, benchmark, summary["n_records"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-model batch runner")
    parser.add_argument("models", nargs="+", help="Registered model name(s) to run")
    parser.add_argument("--benchmark", default="cv-screening", help="Benchmark name")
    parser.add_argument("--output-dir", default="results", help="Base output directory")
    parser.add_argument("--timeout", type=int, default=1800, help="Per-model timeout in seconds")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    for model_name in args.models:
        if model_name not in MODELS:
            logger.error("Unknown model: %s. Available: %s", model_name, ", ".join(MODELS))
            sys.exit(1)
        run_benchmark_for_model(model_name, args.benchmark, args.output_dir, args.timeout)


if __name__ == "__main__":
    main()
