#!/usr/bin/env bash
# Phase 1: Run stereoset + demographic-bias for all 11 models sequentially
# Kill-safe: skips already-completed pairs
set -euo pipefail

cd "$(dirname "$0")/.."
LOG="results/phase1.log"
mkdir -p results

echo "=== Phase 1 start: $(date) ===" | tee "$LOG"

models=(
    "smollm-135m"
    "smollm2-135m"
    "smollm-360m"
    "smollm2-360m"
    "gemma3-270m"
    "qwen25-05b"
    "qwen3-06b"
    "qwen35-08b"
    "lfm2-350m"
    "lfm2-700m"
    "granite4-350m"
)
benchmarks=("stereoset" "demographic-bias")

for model in "${models[@]}"; do
    for bench in "${benchmarks[@]}"; do
        outfile="results/$model/$bench/results.json"
        if [ -f "$outfile" ]; then
            echo "SKIP $model/$bench (exists)" | tee -a "$LOG"
            continue
        fi
        echo "RUN $model/$bench at $(date)" | tee -a "$LOG"
        if uv run python -m slm_bias_testing.runner "$model" --benchmark "$bench" --output-dir results --timeout 3600 2>&1 | tee -a "$LOG"; then
            echo "DONE $model/$bench at $(date)" | tee -a "$LOG"
        else
            echo "FAIL $model/$bench at $(date) — continuing" | tee -a "$LOG"
        fi
        # Let Ollama stabilise between runs
        sleep 10
    done
done

echo "=== Phase 1 complete: $(date) ===" | tee -a "$LOG"
echo "Results:" | tee -a "$LOG"
find results -name "results.json" | sort | tee -a "$LOG"
