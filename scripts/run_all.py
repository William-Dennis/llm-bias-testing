#!/usr/bin/env python3
"""Run all benchmarks for all models. Kill-safe: skips completed pairs."""
import json
import logging
import os
import sys
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    handlers=[logging.FileHandler("results/full_run.log", mode='a'),
                              logging.StreamHandler()])
log = logging.getLogger(__name__)

RESULTS = os.path.expanduser("~/repos/slm-bias-testing/results")
os.chdir(os.path.expanduser("~/repos/slm-bias-testing"))

MODELS = [
    "smollm-135m", "smollm2-135m", "smollm-360m", "smollm2-360m",
    "gemma3-270m", "qwen25-05b", "qwen3-06b", "qwen35-08b",
    "lfm2-350m", "lfm2-700m", "granite4-350m",
]

# All 4 benchmarks
ALL_BENCHMARKS = ["stereoset", "demographic-bias", "winobias", "cv-screening"]

def run_one(model, benchmark):
    outfile = f"{RESULTS}/{model}/{benchmark}/results.json"
    if os.path.exists(outfile):
        log.info("SKIP %s/%s (exists)", model, benchmark)
        return True
    log.info("RUN %s/%s at %s", model, benchmark, datetime.now())
    from slm_bias_testing.runner import run_benchmark_for_model, restart_ollama_clean
    try:
        # Clean restart Ollama before each model to ensure fresh state
        restart_ollama_clean()
        run_benchmark_for_model(model, benchmark, RESULTS, 7200)  # 2h timeout
        ok = os.path.exists(outfile)
        if ok:
            log.info("DONE %s/%s at %s", model, benchmark, datetime.now())
        else:
            log.error("FAIL %s/%s — no results.json after run", model, benchmark)
        return ok
    except Exception as e:
        log.exception("FAIL %s/%s: %s", model, benchmark, e)
        return False

def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    # Parse --num-ctx from argv if present
    num_ctx = None
    for i, arg in enumerate(sys.argv):
        if arg == "--num-ctx" and i + 1 < len(sys.argv):
            num_ctx = int(sys.argv[i + 1])
            os.environ["SLM_NUM_CTX"] = str(num_ctx)

    if num_ctx is not None:
        log.info("Using num_ctx=%d (Ollama 0.19+ MLX cache optimisation)", num_ctx)
    
    if which == "all":
        benchmarks = ALL_BENCHMARKS
    elif which == "phase1":
        benchmarks = ["stereoset", "demographic-bias"]
    elif which == "phase2":
        benchmarks = ["winobias"]
    elif which == "phase3":
        benchmarks = ["cv-screening"]
    else:
        benchmarks = [which]
    
    log.info("=== Run %s at %s ===", which, datetime.now())
    log.info("Models: %d, Benchmarks: %s", len(MODELS), benchmarks)
    
    done = failed = skipped = 0
    total = len(MODELS) * len(benchmarks)
    
    for model in MODELS:
        for benchmark in benchmarks:
            outfile = f"{RESULTS}/{model}/{benchmark}/results.json"
            if os.path.exists(outfile):
                skipped += 1
                continue
            
            if run_one(model, benchmark):
                done += 1
            else:
                failed += 1
            time.sleep(5)
    
    log.info("=== Complete: %d done, %d failed, %d skipped of %d ===", done, failed, skipped, total)
    
    # Summary
    log.info("=== Results summary ===")
    for model in MODELS:
        for benchmark in benchmarks:
            outfile = f"{RESULTS}/{model}/{benchmark}/results.json"
            if os.path.exists(outfile):
                with open(outfile) as f:
                    data = json.load(f)
                n = data.get("n_records") or data.get("n_examples") or 0
                score = data.get("overall_stereotype_score") or data.get("bias_score") or data.get("mean_score") or "-"
                log.info("  %s/%s: n=%d, score=%s", model, benchmark, n, score)

if __name__ == "__main__":
    main()
