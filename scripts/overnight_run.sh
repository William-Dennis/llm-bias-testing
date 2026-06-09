#!/usr/bin/env bash
# Overnight evaluation run for SLM Bias Testing
# Run: bash scripts/overnight_run.sh
# Designed to be kill-safe: re-running skips already-completed model/benchmark pairs.
set -euo pipefail

cd "$(dirname "$0")/.."
RESULTS_DIR="results"
LOG_FILE="${RESULTS_DIR}/run.log"
TIMEOUT=1800  # 30 min per model/benchmark pair

mkdir -p "$RESULTS_DIR"

echo "=== Overnight Eval $(date) ===" | tee -a "$LOG_FILE"
echo "Results dir: $RESULTS_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 1. Ensure Ollama is running
echo "--- Checking Ollama ---" | tee -a "$LOG_FILE"
if ! ollama list >/dev/null 2>&1; then
    echo "Ollama not running. Starting..." | tee -a "$LOG_FILE"
    ollama serve &
    sleep 3
    if ! ollama list >/dev/null 2>&1; then
        echo "ERROR: Failed to start Ollama" | tee -a "$LOG_FILE"
        exit 1
    fi
fi
echo "Ollama OK" | tee -a "$LOG_FILE"

# 2. Pull all models (download first so eval time isn't eaten by downloads)
echo "" | tee -a "$LOG_FILE"
echo "--- Pulling models ---" | tee -a "$LOG_FILE"
UNDER_1B_MODELS=(
    "smollm-135m:smollm:135m"
    "smollm-360m:smollm:360m"
    "qwen25-05b:qwen2.5:0.5b"
    "smollm2-135m:smollm2:135m"
    "smollm2-360m:smollm2:360m"
    "gemma3-270m:gemma3:270m"
    "qwen3-06b:qwen3:0.6b"
    "qwen35-08b:qwen3.5:0.8b"
    "lfm2-350m:sam860/lfm2:350m"
    "lfm2-700m:sam860/lfm2:700m"
    "granite4-350m:granite4:350m"
)
for entry in "${UNDER_1B_MODELS[@]}"; do
    tag="${entry#*:}"
    echo "  Pulling $tag ..." | tee -a "$LOG_FILE"
    ollama pull "$tag" 2>&1 | tail -1 | tee -a "$LOG_FILE"
done
echo "All models pulled" | tee -a "$LOG_FILE"

# 3. Run benchmarks
echo "" | tee -a "$LOG_FILE"
echo "--- Running benchmarks ---" | tee -a "$LOG_FILE"

# Build model list for runner (registered names only)
MODEL_NAMES=""
for entry in "${UNDER_1B_MODELS[@]}"; do
    name="${entry%%:*}"
    if [ -z "$MODEL_NAMES" ]; then
        MODEL_NAMES="$name"
    else
        MODEL_NAMES="$MODEL_NAMES,$name"
    fi
done

echo "Models: $MODEL_NAMES" | tee -a "$LOG_FILE"
echo "Timeout per pair: ${TIMEOUT}s" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Phase 1: Light benchmarks (stereoset + demographic-bias)
echo "=== Phase 1: Light benchmarks (stereoset + demographic-bias) ===" | tee -a "$LOG_FILE"
uv run python scripts/run_experiments.py \
    --models "$MODEL_NAMES" \
    --benchmarks "stereoset,demographic-bias" \
    --output-dir "$RESULTS_DIR" \
    --timeout "$TIMEOUT" 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "=== Phase 2: WinoBias ===" | tee -a "$LOG_FILE"
uv run python scripts/run_experiments.py \
    --models "$MODEL_NAMES" \
    --benchmarks "winobias" \
    --output-dir "$RESULTS_DIR" \
    --timeout "$TIMEOUT" 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "=== Phase 3: CV screening (100 CVs x 10 runs) ===" | tee -a "$LOG_FILE"
for entry in "${UNDER_1B_MODELS[@]}"; do
    name="${entry%%:*}"
    echo "  CV screening: $name" | tee -a "$LOG_FILE"
    uv run python -m slm_bias_testing.runner \
        "$name" \
        --benchmark cv-screening \
        --output-dir "$RESULTS_DIR" \
        --timeout "$TIMEOUT" \
        --max-samples 100 2>&1 | tee -a "$LOG_FILE"
done

# 4. Generate temporal analysis
echo "" | tee -a "$LOG_FILE"
echo "=== Phase 4: Temporal analysis ===" | tee -a "$LOG_FILE"
uv run python -m slm_bias_testing.temporal \
    --results-dir "$RESULTS_DIR" 2>&1 | tee -a "$LOG_FILE"

# 5. Summary
echo "" | tee -a "$LOG_FILE"
echo "=== Summary ===" | tee -a "$LOG_FILE"
echo "Completed: $(date)" | tee -a "$LOG_FILE"
echo "Results in: $RESULTS_DIR" | tee -a "$LOG_FILE"

# Print result counts
echo "" | tee -a "$LOG_FILE"
echo "Result files:" | tee -a "$LOG_FILE"
find "$RESULTS_DIR" -name "results.json" | sort | while read f; do
    echo "  $f" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "Done." | tee -a "$LOG_FILE"
