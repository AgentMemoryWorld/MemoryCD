#!/bin/bash
# Parallel single-domain evaluation driver for MemoryCD.
#
# Loops over: DOMAINS x TASKS x METHODS x MODELS and runs evaluation_all_4task.py
# in parallel (uses GNU `parallel` if available, otherwise background jobs).
#
# Usage:  ./run_evaluation_pa.sh [num_parallel]
# Example: ./run_evaluation_pa.sh 4

set -euo pipefail

# --- Required environment ---------------------------------------------------
: "${API_KEY:?API_KEY env var is required (export API_KEY=...)}"

# Map API_KEY into whichever provider variable the user has configured.
export OPENAI_API_KEY="${OPENAI_API_KEY:-$API_KEY}"

BASE_DIR="${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
INPUT="${INPUT:-$BASE_DIR/data/cross_domain.jsonl}"
META_DIR="${META_DIR:-$BASE_DIR/meta}"
NUM_PARALLEL="${1:-2}"

# --- Sweep configuration ----------------------------------------------------
DOMAINS=(
    "Beauty_and_Personal_Care"
    "Books"
    "Electronics"
    "Home_and_Kitchen"
)

TASKS=(
    "rating_prediction"
    "review_summarization"
    "review_generation"
    "item_ranking"
)

# Format: "method:max_memory_items" (use "all" for no cap)
METHODS=(
    "long_context:all"
    "rag:10"
)

MODELS=(
    "openai/gpt-5"
    # "openai/gpt-5-mini"
    # "anthropic/claude-sonnet-4"
)

# Optional quick-test overrides (leave unset for full runs)
# MAX_USERS=5
# NUM_TEST=3

# --- Plumbing ---------------------------------------------------------------
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; RED='\033[0;31m'; NC='\033[0m'
TEMP_DIR="$BASE_DIR/temp_eval_$$"
mkdir -p "$TEMP_DIR"

run_one() {
    local task="$1"
    local method_cfg="$2"
    local model="$3"
    local domain="$4"

    local method="${method_cfg%%:*}"
    local mem_items="${method_cfg##*:}"
    local model_safe="${model//\//_}"
    local log="$TEMP_DIR/${domain}_${task}_${method}_${model_safe}.log"
    local status="$TEMP_DIR/${domain}_${task}_${method}_${model_safe}.status"

    echo -e "${YELLOW}[$(date +%H:%M:%S)] Start: $domain | $task | $method | $model${NC}"

    local cmd="python $BASE_DIR/evaluation_all_4task.py \
        --task $task --method $method --domain $domain \
        --input $INPUT --meta-dir $META_DIR \
        --llm-model $model --log-dir $BASE_DIR/logs/single"
    if [ "$mem_items" != "all" ]; then
        cmd="$cmd --max-memory-items $mem_items"
    fi
    [ -n "${MAX_USERS:-}" ] && cmd="$cmd --max-users $MAX_USERS"
    [ -n "${NUM_TEST:-}" ] && cmd="$cmd --num-test $NUM_TEST"

    if eval "$cmd" > "$log" 2>&1; then
        echo SUCCESS > "$status"
        echo -e "${GREEN}[$(date +%H:%M:%S)] OK:    $domain | $task | $method | $model${NC}"
    else
        echo FAILED > "$status"
        echo -e "${RED}[$(date +%H:%M:%S)] FAIL:  $domain | $task | $method | $model (log: $log)${NC}"
    fi
}
export -f run_one
export API_KEY OPENAI_API_KEY BASE_DIR INPUT META_DIR TEMP_DIR MAX_USERS NUM_TEST
export GREEN YELLOW BLUE RED NC

COMBOS=()
for d in "${DOMAINS[@]}"; do
    for t in "${TASKS[@]}"; do
        for m in "${METHODS[@]}"; do
            for mo in "${MODELS[@]}"; do
                COMBOS+=("$t|$m|$mo|$d")
            done
        done
    done
done

echo "==========================================================="
echo "MemoryCD single-domain parallel evaluation"
echo "  Domains: ${#DOMAINS[@]}  Tasks: ${#TASKS[@]}  Methods: ${#METHODS[@]}  Models: ${#MODELS[@]}"
echo "  Combinations: ${#COMBOS[@]}   Parallel: $NUM_PARALLEL"
echo "  Input: $INPUT"
echo "==========================================================="

START=$(date +%s)
if command -v parallel >/dev/null 2>&1; then
    printf '%s\n' "${COMBOS[@]}" | parallel -j "$NUM_PARALLEL" --colsep '\|' run_one {1} {2} {3} {4}
else
    running=0
    for combo in "${COMBOS[@]}"; do
        IFS='|' read -r t m mo d <<< "$combo"
        while [ "$running" -ge "$NUM_PARALLEL" ]; do
            sleep 1
            running=$(jobs -r | wc -l)
        done
        run_one "$t" "$m" "$mo" "$d" &
        running=$((running + 1))
    done
    wait
fi
END=$(date +%s)

ok=0; fail=0
for combo in "${COMBOS[@]}"; do
    IFS='|' read -r t m mo d <<< "$combo"
    method="${m%%:*}"
    model_safe="${mo//\//_}"
    s="$TEMP_DIR/${d}_${t}_${method}_${model_safe}.status"
    if [ -f "$s" ] && [ "$(cat "$s")" = SUCCESS ]; then ok=$((ok+1)); else fail=$((fail+1)); fi
done

echo "==========================================================="
echo -e "${BLUE}Done.${NC}  Total: ${#COMBOS[@]}  ${GREEN}OK: $ok${NC}  ${RED}Fail: $fail${NC}  Time: $((END-START))s"
echo "Per-job logs: $TEMP_DIR"
echo "Eval outputs: $BASE_DIR/logs/single/<domain>/"
echo "==========================================================="
