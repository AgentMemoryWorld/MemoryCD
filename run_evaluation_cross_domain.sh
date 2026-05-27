#!/bin/bash
# Parallel cross-domain evaluation driver for MemoryCD.
#
# Loops over: TARGETS x SOURCE_CONFIGS x TASKS x METHODS x MODELS
# and runs evaluation_all_4task_cross_domain.py in parallel.
#
# Usage:  ./run_evaluation_cross_domain.sh [num_parallel]
# Example: ./run_evaluation_cross_domain.sh 4

set -euo pipefail

: "${API_KEY:?API_KEY env var is required (export API_KEY=...)}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-$API_KEY}"

BASE_DIR="${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
INPUT="${INPUT:-$BASE_DIR/data/cross_domain.jsonl}"
META_DIR="${META_DIR:-$BASE_DIR/meta}"
NUM_PARALLEL="${1:-2}"

# --- Sweep configuration ----------------------------------------------------
TARGET_DOMAINS=(
    "Beauty_and_Personal_Care"
    "Books"
    "Electronics"
    "Home_and_Kitchen"
)

# Format: "label:extra-flags". Empty flags means "all non-target domains".
SOURCE_CONFIGS=(
    "all:"
    # "none:--no-memory"
    # "Books:--source-domains Books"
    # "BooksElec:--source-domains Books Electronics"
)

TASKS=(
    "rating_prediction"
    "review_summarization"
    "review_generation"
    "item_ranking"
)

METHODS=(
    "long_context:all"
    "rag:10"
)

MODELS=(
    "openai/gpt-5"
)

# Optional quick-test overrides
# MAX_USERS=5
# NUM_TEST=3

# --- Plumbing ---------------------------------------------------------------
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; RED='\033[0;31m'; NC='\033[0m'
TEMP_DIR="$BASE_DIR/temp_eval_cross_domain_$$"
mkdir -p "$TEMP_DIR"

run_one() {
    local task="$1"
    local method_cfg="$2"
    local model="$3"
    local target="$4"
    local label="$5"
    local extra="$6"

    local method="${method_cfg%%:*}"
    local mem_items="${method_cfg##*:}"
    local model_safe="${model//\//_}"
    local key="${target}_${label}_${task}_${method}_${model_safe}"
    local log="$TEMP_DIR/${key}.log"
    local status="$TEMP_DIR/${key}.status"

    echo -e "${YELLOW}[$(date +%H:%M:%S)] Start: $target/$label | $task | $method | $model${NC}"

    local cmd="python $BASE_DIR/evaluation_all_4task_cross_domain.py \
        --task $task --method $method --target-domain $target \
        --input $INPUT --meta-dir $META_DIR \
        --llm-model $model"
    if [ -n "$extra" ]; then
        cmd="$cmd $extra"
    fi
    if [ "$mem_items" != "all" ]; then
        cmd="$cmd --max-memory-items $mem_items"
    fi
    [ -n "${MAX_USERS:-}" ] && cmd="$cmd --max-users $MAX_USERS"
    [ -n "${NUM_TEST:-}" ] && cmd="$cmd --num-test $NUM_TEST"

    if eval "$cmd" > "$log" 2>&1; then
        echo SUCCESS > "$status"
        echo -e "${GREEN}[$(date +%H:%M:%S)] OK:    $target/$label | $task | $method | $model${NC}"
    else
        echo FAILED > "$status"
        echo -e "${RED}[$(date +%H:%M:%S)] FAIL:  $target/$label | $task | $method | $model (log: $log)${NC}"
    fi
}
export -f run_one
export API_KEY OPENAI_API_KEY BASE_DIR INPUT META_DIR TEMP_DIR MAX_USERS NUM_TEST
export GREEN YELLOW BLUE RED NC

COMBOS=()
for tgt in "${TARGET_DOMAINS[@]}"; do
    for cfg in "${SOURCE_CONFIGS[@]}"; do
        label="${cfg%%:*}"
        extra="${cfg#*:}"
        for t in "${TASKS[@]}"; do
            for m in "${METHODS[@]}"; do
                for mo in "${MODELS[@]}"; do
                    COMBOS+=("$t|$m|$mo|$tgt|$label|$extra")
                done
            done
        done
    done
done

echo "==========================================================="
echo "MemoryCD cross-domain parallel evaluation"
echo "  Targets: ${#TARGET_DOMAINS[@]}  Sources: ${#SOURCE_CONFIGS[@]}  Tasks: ${#TASKS[@]}  Methods: ${#METHODS[@]}  Models: ${#MODELS[@]}"
echo "  Combinations: ${#COMBOS[@]}   Parallel: $NUM_PARALLEL"
echo "  Input: $INPUT"
echo "==========================================================="

START=$(date +%s)
if command -v parallel >/dev/null 2>&1; then
    printf '%s\n' "${COMBOS[@]}" | parallel -j "$NUM_PARALLEL" --colsep '\|' run_one {1} {2} {3} {4} {5} {6}
else
    running=0
    for combo in "${COMBOS[@]}"; do
        IFS='|' read -r t m mo tgt label extra <<< "$combo"
        while [ "$running" -ge "$NUM_PARALLEL" ]; do
            sleep 1
            running=$(jobs -r | wc -l)
        done
        run_one "$t" "$m" "$mo" "$tgt" "$label" "$extra" &
        running=$((running + 1))
    done
    wait
fi
END=$(date +%s)

ok=0; fail=0
for combo in "${COMBOS[@]}"; do
    IFS='|' read -r t m mo tgt label extra <<< "$combo"
    method="${m%%:*}"
    model_safe="${mo//\//_}"
    s="$TEMP_DIR/${tgt}_${label}_${t}_${method}_${model_safe}.status"
    if [ -f "$s" ] && [ "$(cat "$s")" = SUCCESS ]; then ok=$((ok+1)); else fail=$((fail+1)); fi
done

echo "==========================================================="
echo -e "${BLUE}Done.${NC}  Total: ${#COMBOS[@]}  ${GREEN}OK: $ok${NC}  ${RED}Fail: $fail${NC}  Time: $((END-START))s"
echo "Per-job logs: $TEMP_DIR"
echo "Eval outputs: $BASE_DIR/logs/cross/<target_domain>/"
echo "==========================================================="
