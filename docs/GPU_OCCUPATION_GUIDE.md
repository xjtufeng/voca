# GPU Occupation Guide

Simple and stable GPU occupation to prevent resource reclamation on HPC.

## Quick Start

### Option 1: Direct Python Script

**Foreground (testing)**:
```bash
python occupy_gpu.py --gpu 0 --percent 30
```

**Background (recommended)**:
```bash
nohup python occupy_gpu.py --gpu 0 --percent 30 --quiet > gpu_occupy.log 2>&1 &
```

### Option 2: Helper Scripts

**Start occupation**:
```bash
cd ~/jhspoolers/voca
bash scripts/start_gpu_occupy.sh 0 30
# Args: GPU_ID (default 0), PERCENT (default 30)
```

**Stop occupation**:
```bash
bash scripts/stop_gpu_occupy.sh 0
# Args: GPU_ID (default 0)
```

## Arguments

- `--gpu`: GPU device ID (default: 0)
- `--percent`: Target utilization percentage, 10-90 (default: 30)
- `--quiet`: Suppress output for background running

## Examples

**Occupy 30% of GPU 0**:
```bash
python occupy_gpu.py --gpu 0 --percent 30
```

**Occupy 50% of GPU 1 in background**:
```bash
nohup python occupy_gpu.py --gpu 1 --percent 50 --quiet > gpu1.log 2>&1 &
```

**Multiple GPUs**:
```bash
bash scripts/start_gpu_occupy.sh 0 30  # GPU 0 at 30%
bash scripts/start_gpu_occupy.sh 1 40  # GPU 1 at 40%
bash scripts/start_gpu_occupy.sh 2 30  # GPU 2 at 30%
```

## Monitoring

Check GPU status:
```bash
nvidia-smi
# or watch in real-time
watch -n 1 nvidia-smi
```

Find running processes:
```bash
ps aux | grep occupy_gpu.py
```

Check logs:
```bash
tail -f gpu_occupy_0.log
```

## Stopping

**Using script**:
```bash
bash scripts/stop_gpu_occupy.sh 0
```

**Manual kill**:
```bash
# Find PID
ps aux | grep occupy_gpu.py

# Kill process
kill <PID>
# or force kill
kill -9 <PID>
```

**Kill all occupation processes**:
```bash
pkill -f occupy_gpu.py
```

## Technical Details

- **Memory usage**: ~200-500 MB per GPU
- **Computation**: Continuous forward/backward passes through lightweight neural network
- **Stability**: Self-adjusting computation intensity based on target percentage
- **No interference**: Runs independently, won't block other training jobs

## Expected nvidia-smi Output

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.x   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  On   | 00000000:00:1E.0 Off |                    0 |
| N/A   42C    P0    78W / 300W |    345MiB / 16384MiB |     32%      Default |
+-------------------------------+----------------------+----------------------+
```

GPU-Util should be around **30-40%** for `--percent 30`.

