#!/bin/bash
# Start vllm server and run evaluation

set -e


# ===== Parameters =====
START_VLLM="${START_VLLM:-false}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-test}"
NUM_GAMES="${NUM_GAMES:-1}"

# ===== Configuration =====
MODEL_PATH="${VLLM_MODEL_PATH:-/mnt/data/yunpeng.zyp/models/Qwen3-14B}"
HOST="${VLLM_HOST:-localhost}"
PORT="${VLLM_PORT:-8000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$SCRIPT_DIR/configs/task_config.yaml}"

# Ensure CONFIG_FILE is an absolute path
if [[ "$CONFIG_FILE" != /* ]]; then
    CONFIG_FILE="$(cd "$(dirname "$CONFIG_FILE")" && pwd)/$(basename "$CONFIG_FILE")"
fi

# Verify config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE" >&2
    exit 1
fi

MAX_WORKERS="${MAX_WORKERS:-}"

MODEL_NAME="local_model"

# ===== Helper Functions =====
check_server() {
    curl -s "http://${HOST}:${PORT}/v1/models" > /dev/null 2>&1
}

start_vllm_server() {
    echo "Starting vllm server..."
    vllm serve "$MODEL_PATH" \
        --served-model-name "$MODEL_NAME" \
        --host "$HOST" \
        --port "$PORT" > /dev/null 2>&1 &
    
    VLLM_PID=$!
    
    # Wait for server to be ready
    for i in {1..10}; do
        if check_server; then
            echo "Server started successfully"
            return 0
        fi
        [ $i -eq 10 ] && { echo "Error: Server failed to start" >&2; exit 1; }
        sleep 30
    done
}

cleanup() {
    [ -n "$VLLM_PID" ] && kill $VLLM_PID 2>/dev/null && wait $VLLM_PID 2>/dev/null
}

# ===== Main =====
trap cleanup EXIT INT TERM

# Start server if needed
if [ "$START_VLLM" = "true" ] && ! check_server; then
    start_vllm_server
fi

# Run evaluation
cd "$SCRIPT_DIR"
python run_eval.py \
    --config "$CONFIG_FILE" \
    --num-games "$NUM_GAMES" \
    ${MAX_WORKERS:+--max-workers "$MAX_WORKERS"} \
    ${EXPERIMENT_NAME:+--experiment-name "$EXPERIMENT_NAME"} \
    "$@"
