import argparse
import logging
import os

from llm_bias_testing.benchmark import (
    run_benchmark,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Bias Testing Benchmarks")
    parser.add_argument("--model", default="gemma3:1b-it-qat", help="Model name to use")
    parser.add_argument("--models", help="Comma-separated list of model names")
    parser.add_argument("--timeout", type=int, default=1800, help="Per-model timeout in seconds")
    parser.add_argument("--output-dir", default="results", help="Results directory")
    parser.add_argument(
        "--benchmark",
        default="cv-screening",
        choices=["cv-screening", "stereoset", "crows-pairs", "bbq", "all"],
        help="Benchmark to run (default: cv-screening)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        models = [args.model]

    if args.benchmark == "cv-screening":
        for model_name in models:
            model_output_dir = os.path.join(args.output_dir, model_name.replace(":", "-"))
            os.makedirs(model_output_dir, exist_ok=True)
            run_benchmark(model_name, model_output_dir, args.timeout)
    else:
        from llm_bias_testing.runner import BENCHMARK_CHOICES, get_benchmarks, run_benchmark_for_model

        bench_list = get_benchmarks(args.benchmark)
        for model_name in models:
            for bench in bench_list:
                run_benchmark_for_model(model_name, bench, args.output_dir, args.timeout)


if __name__ == "__main__":
    main()
