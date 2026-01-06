#!/bin/bash
# Stop GPU occupation process

GPU_ID=${1:-0}
PID_FILE="gpu_occupy_${GPU_ID}.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat $PID_FILE)
    echo "Stopping GPU occupation process: PID=$PID"
    kill $PID 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo "Process stopped successfully"
        rm $PID_FILE
    else
        echo "Process not found or already stopped"
        rm $PID_FILE
    fi
else
    echo "PID file not found: $PID_FILE"
    echo "Looking for occupy_gpu.py processes..."
    ps aux | grep "occupy_gpu.py" | grep -v grep
    echo ""
    echo "To stop manually: kill <PID>"
fi


