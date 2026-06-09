import argparse
import logging
import os

from slm_bias_testing.benchmark import run_benchmark
from slm_bias_testing.registry import MODELS, get_model

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="CV Screening Bias Test")
    parser.add_argument("--model", default="gemma3-1b", help="Registered model name (e.g. gemma3-1b)")
    parser.add_argument("--models", help="Comma-separated registered model names")
    parser.add_argument("--timeout", type=int, default=1800, help="Per-model timeout in seconds")
    parser.add_argument("--output-dir", default="results", help="Results directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.models:
        model_names = [m.strip() for m in args.models.split(",")]
    else:
        model_names = [args.model]

    # Validate all model names upfront
    invalid = [m for m in model_names if m not in MODELS]
    if invalid:
        logger.error("Unknown model(s): %s. Available: %s", ", ".join(invalid), ", ".join(MODELS))
        return

    for model_name in model_names:
        config = get_model(model_name)
        ollama_tag = config["ollama_tag"]
        model_output_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        run_benchmark(ollama_tag, model_output_dir, args.timeout)


if __name__ == "__main__":
    main()
