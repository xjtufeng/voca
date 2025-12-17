#!/bin/bash
# Start GPU occupation in background

GPU_ID=${1:-0}
PERCENT=${2:-30}

echo "Starting GPU occupation..."
echo "GPU: $GPU_ID, Target: ${PERCENT}%"

cd ~/jhspoolers/voca
source ~/.bashrc
conda activate voca

nohup python occupy_gpu.py --gpu $GPU_ID --percent $PERCENT --quiet > gpu_occupy_${GPU_ID}.log 2>&1 &

PID=$!
echo "Process started: PID=$PID"
echo "Log file: gpu_occupy_${GPU_ID}.log"
echo "Stop with: kill $PID"

echo $PID > gpu_occupy_${GPU_ID}.pid
echo "PID saved to: gpu_occupy_${GPU_ID}.pid"

