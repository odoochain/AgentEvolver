#!/bin/bash
# Start vllm server and run diplomacy evaluation

set -e  # Exit on error

# Configuration (edit MODEL_PATH if you want to start local vLLM)
MODEL_PATH="/mnt/data_aisys_cpfs/xielipeng.xlp/models/Qwen2.5-7B-Instruct"
HOST="localhost"
PORT=8000
MODEL_NAME="local_model"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/task_config.yaml"

# Function to check if server is already running
check_server() {
    if curl -s "http://${HOST}:${PORT}/v1/models" > /dev/null 2>&1; then
        return 0  # Server is running
    else
        return 1  # Server is not running
    fi
}

# Function to cleanup on exit
cleanup() {
    if [ -n "$VLLM_PID" ] && [ "$SERVER_STARTED" = "true" ]; then
        echo "Stopping vllm server (PID: $VLLM_PID)..."
        kill $VLLM_PID 2>/dev/null || true
        wait $VLLM_PID 2>/dev/null || true
    fi
}

# Register cleanup function
trap cleanup EXIT INT TERM

echo "Checking if vllm server is already running on ${HOST}:${PORT}..."
if check_server; then
    echo "✓ vllm server is already running on ${HOST}:${PORT}"
    SERVER_STARTED="false"
    VLLM_PID=""
else
    echo "vllm server is not running. Starting vllm server..."
    SERVER_STARTED="true"
    vllm serve "$MODEL_PATH" \
        --served-model-name "$MODEL_NAME" \
        --host "$HOST" \
        --port "$PORT" &

    VLLM_PID=$!

    # Wait for server to be ready
    echo "Waiting for vllm server to be ready..."
    MAX_ATTEMPTS=10
    WAIT_INTERVAL=30

    for i in $(seq 1 $MAX_ATTEMPTS); do
        if check_server; then
            echo "✓ vllm server is ready"
            break
        fi
        if [ $i -eq $MAX_ATTEMPTS ]; then
            echo "Error: vllm server failed to start within 5 minutes"
            exit 1
        fi
        echo "Waiting... (attempt $i/$MAX_ATTEMPTS, will wait ${WAIT_INTERVAL}s)"
        sleep $WAIT_INTERVAL
    done
fi

echo "Running diplomacy evaluation..."
cd "$SCRIPT_DIR"
python run_eval.py --config "$CONFIG_FILE"

