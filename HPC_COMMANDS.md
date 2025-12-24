# HPC Training Commands - Quick Reference

快速参考：在 HPC 上训练三分支模型的所有命令。

---

## 🚀 第一步：准备环境

```bash
# 登录 HPC
ssh xfeng733@hpc2login.hpc.hkust-gz.edu.cn

# 进入项目目录
cd ~/jhspoolers/voca

# 激活环境
conda activate voca

# 拉取最新代码
git pull

# 检查数据
ls -lh /hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_feats/real/ | head -5
ls -lh /hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_feats/fake/ | head -5

# 统计视频数量
find /hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_feats -name "visual_embeddings.npz" | wc -l
```

---

## ⚡ 第二步：快速测试（5 epochs，约 30 分钟）

```bash
# 给脚本执行权限
chmod +x scripts/quick_test_three_branch.sh

# 运行快速测试
bash scripts/quick_test_three_branch.sh

# 在另一个终端监控
tail -f logs/quick_test_*.log

# 检查结果
ls -lh checkpoints/quick_test/
```

**预期输出：**
```
Epoch [5/5]
  Val   - Fused AUC: 0.8890, CM AUC: 0.8656, AO AUC: 0.7923, VO AUC: 0.8434
Quick test completed!
```

---

## 🎯 第三步：完整训练（100 epochs）

### 方式 A：使用 tmux（推荐）

```bash
# 创建 tmux 会话
tmux new -s train

# 在 tmux 中运行训练
conda activate voca
bash scripts/train_three_branch_baseline.sh

# 分离会话：按 Ctrl+B 然后按 D
# 查看所有会话
tmux ls

# 重新连接
tmux attach -t train

# 杀死会话（如果需要）
tmux kill-session -t train
```

### 方式 B：使用 nohup

```bash
# 后台运行
nohup bash scripts/train_three_branch_baseline.sh > train.log 2>&1 &

# 查看进程
ps aux | grep train_three_branch.py

# 查看日志
tail -f train.log

# 或查看最新的训练日志
tail -f logs/train_baseline_*.log
```

### 方式 C：直接运行（简单但需要保持连接）

```bash
# 给脚本执行权限
chmod +x scripts/train_three_branch_baseline.sh

# 直接运行
bash scripts/train_three_branch_baseline.sh
```

---

## 📊 第四步：监控训练

### 实时查看日志

```bash
# 查看最新的训练日志
tail -f logs/train_baseline_*.log

# 只看关键指标
grep "Val   - Fused AUC" logs/train_baseline_*.log

# 查看最佳模型
grep "New best" logs/train_baseline_*.log
```

### 监控 GPU

```bash
# 实时监控
watch -n 2 nvidia-smi

# 查看 GPU 利用率
nvidia-smi dmon -s u -i 0
```

### 检查 checkpoints

```bash
# 查看已保存的模型
ls -lht checkpoints/three_branch_baseline/

# 查看最佳模型的指标
python -c "
import torch
ckpt = torch.load('checkpoints/three_branch_baseline/best.pth', map_location='cpu')
print('Epoch:', ckpt['epoch'])
print('Metrics:', ckpt['metrics'])
"
```

---

## 🎛️ 可选：自定义训练参数

### 直接使用 Python 命令（不用脚本）

```bash
# 基础训练
python train_three_branch.py \
  --features_root /hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_feats \
  --splits train dev \
  --batch_size 16 \
  --epochs 100 \
  --lr 1e-3 \
  --max_frames 150 \
  --d_model 512 \
  --nhead 8 \
  --cm_layers 4 \
  --ao_layers 3 \
  --vo_layers 3 \
  --fusion_method weighted \
  --output_dir checkpoints/baseline \
  --save_every 5 \
  --num_workers 4 \
  2>&1 | tee logs/my_training.log
```

### 尝试不同的 fusion 方法

```bash
# Concat fusion
python train_three_branch.py \
  --features_root /hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_feats \
  --fusion_method concat \
  --output_dir checkpoints/fusion_concat \
  ... (其他参数)

# Attention fusion
python train_three_branch.py \
  --features_root /hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_feats \
  --fusion_method attention \
  --output_dir checkpoints/fusion_attention \
  ... (其他参数)
```

### 调整 loss weights

```bash
# 增加 Audio-Only 分支的权重
python train_three_branch.py \
  --features_root /hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_feats \
  --ao_loss_weight 0.5 \
  --output_dir checkpoints/ao_weighted \
  ... (其他参数)
```

---

## 🔧 故障排除命令

### 检查环境

```bash
# Python 版本
python --version

# PyTorch 版本和 CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 检查依赖
python -c "import numpy, sklearn, tqdm; print('All deps OK')"
```

### 检查数据

```bash
# 检查特征文件
python -c "
import numpy as np
import glob

visual_files = glob.glob('/hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_feats/real/*/visual_embeddings.npz')
print(f'Found {len(visual_files)} visual files')

sample = np.load(visual_files[0])
print(f'Sample shape: {sample[\"embeddings\"].shape}')
"
```

### 查找并杀死训练进程

```bash
# 查找进程
ps aux | grep train_three_branch.py

# 杀死进程（如果需要）
kill -9 <PID>

# 或杀死所有 Python 进程（谨慎！）
pkill -9 python
```

### 清理磁盘空间

```bash
# 查看磁盘使用
df -h

# 查看项目大小
du -sh ~/jhspoolers/voca
du -sh ~/jhspoolers/voca/checkpoints
du -sh ~/jhspoolers/voca/logs

# 删除旧的 checkpoints（谨慎！）
rm -rf checkpoints/old_experiment_*
```

---

## 📈 训练后的命令

### 评估模型

```bash
# 加载最佳模型并评估
python -c "
import torch
from model_three_branch import ThreeBranchJointModel

ckpt = torch.load('checkpoints/three_branch_baseline/best.pth', map_location='cpu')
print('Best model metrics:')
for k, v in ckpt['metrics'].items():
    if 'auc' in k or 'f1' in k:
        print(f'  {k}: {v:.4f}')
"
```

### 下载模型到本地

```bash
# 在本地机器上执行
scp -r xfeng733@hpc2login.hpc.hkust-gz.edu.cn:~/jhspoolers/voca/checkpoints/three_branch_baseline ./
```

### 上传新代码到 HPC

```bash
# 在本地机器上执行
scp new_script.py xfeng733@hpc2login.hpc.hkust-gz.edu.cn:~/jhspoolers/voca/

# 或使用 git
# 本地: git push
# HPC: git pull
```

---

## 🎉 完整的训练流程（一键复制）

```bash
# ============================================
# 完整的三分支模型训练流程
# ============================================

# 1. 登录和准备
ssh xfeng733@hpc2login.hpc.hkust-gz.edu.cn
cd ~/jhspoolers/voca
conda activate voca
git pull

# 2. 快速测试（确保一切正常）
bash scripts/quick_test_three_branch.sh
# 等待约 30 分钟，检查输出

# 3. 正式训练（在 tmux 中）
tmux new -s train
conda activate voca
bash scripts/train_three_branch_baseline.sh

# 4. 分离 tmux（Ctrl+B 然后 D）

# 5. 在另一个终端监控
tail -f logs/train_baseline_*.log

# 6. 定期检查
tmux attach -t train  # 重新连接
nvidia-smi  # 查看 GPU
ls -lh checkpoints/three_branch_baseline/  # 查看 checkpoints

# 完成！预计 24-48 小时后训练完成
```

---

## ⏱️ 预计时间

| 任务 | 时间 | GPU | 说明 |
|------|------|-----|------|
| **快速测试** | ~30 min | A800 | 5 epochs, 小模型 |
| **完整训练** | 24-48 hours | A800 | 100 epochs, 完整模型 |
| **单个 epoch** | ~15-20 min | A800 | 取决于数据量 |

---

## 📞 遇到问题？

```bash
# 查看完整文档
cat docs/TRAINING_GUIDE.md

# 检查模型架构
cat docs/THREE_BRANCH_GUIDE.md

# 查看 DFD-FCG 集成
cat docs/DFD_FCG_INTEGRATION.md
```

---

好了！现在可以开始训练了 🚀

