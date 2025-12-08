#!/bin/bash
# Start vllm server and run evaluation

set -e  # Exit on error

# Configuration
MODEL_PATH="/mnt/data_aisys_cpfs/xielipeng.xlp/models/Qwen2.5-7B-Instruct"
HOST="localhost"
PORT=8000

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/task_config.yaml"

MODEL_NAME="local_model"

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

# 1. 检查 vllm server 是否已经运行
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

# 2. 运行 run_eval.py，传递 task_config.yaml 路径
echo "Running evaluation..."
cd "$SCRIPT_DIR"
python run_eval.py --config "$CONFIG_FILE"

