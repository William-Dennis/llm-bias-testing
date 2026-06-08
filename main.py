import argparse
import logging
import os

from llm_bias_testing.benchmark import (
    run_benchmark,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="CV Screening Bias Test")
    parser.add_argument("--model", default="gemma3:1b-it-qat", help="Model name to use")
    parser.add_argument("--models", help="Comma-separated list of model names")
    parser.add_argument("--timeout", type=int, default=1800, help="Per-model timeout in seconds")
    parser.add_argument("--output-dir", default="results", help="Results directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        models = [args.model]

    for model_name in models:
        model_output_dir = os.path.join(args.output_dir, model_name.replace(":", "-"))
        os.makedirs(model_output_dir, exist_ok=True)
        run_benchmark(model_name, model_output_dir, args.timeout)


if __name__ == "__main__":
    main()
