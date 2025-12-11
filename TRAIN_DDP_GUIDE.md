# 4卡A800 DDP训练指南

## 已推送到 GitHub
最新代码已推送至：`https://github.com/xjtufeng/voca`
- Commit: `train_crossmodal_ddp.py` (50 epochs, Transformer + Cross-Attention + Contrastive)

---

## HPC 上拉取最新代码

```bash
cd /hpc2hdd/home/xfeng733/jhspoolers/voca
git pull origin main
```

---

## 环境准备

```bash
# 激活环境
conda activate voca

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONNOUSERSITE=1
export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1
```

---

## 训练命令（推荐配置）

### 基础版（4卡，batch=64/卡，50 epochs）

```bash
torchrun --nproc_per_node=4 --master_port=29500 \
  train_crossmodal_ddp.py \
  --features_root /hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_feats \
  --batch_size 64 \
  --epochs 50 \
  --seq_len 256 \
  --hidden 768 \
  --num_layers 4 \
  --num_heads 8 \
  --lr 3e-4 \
  --weight_decay 1e-4 \
  --warmup_epochs 3 \
  --dropout 0.1 \
  --use_contrastive \
  --contrastive_weight 0.5 \
  --label_smoothing 0.1 \
  --num_workers 8 \
  --pin_memory \
  --save_path best_model_4gpu_50ep.pt
```

### 高配版（追求最佳效果，显存充足）

```bash
torchrun --nproc_per_node=4 --master_port=29500 \
  train_crossmodal_ddp.py \
  --features_root /hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_feats \
  --batch_size 80 \
  --epochs 50 \
  --seq_len 320 \
  --hidden 1024 \
  --num_layers 6 \
  --num_heads 16 \
  --lr 2e-4 \
  --weight_decay 1e-4 \
  --warmup_epochs 5 \
  --dropout 0.15 \
  --use_contrastive \
  --contrastive_weight 0.6 \
  --label_smoothing 0.1 \
  --num_workers 8 \
  --pin_memory \
  --save_path best_model_4gpu_50ep_large.pt
```

---

## 监控 GPU

```bash
# 实时监控4卡利用率
watch -n 1 nvidia-smi

# 或单次查看
nvidia-smi
```

---

## 模型架构说明

### CrossModalTransformer
- **投影层**：视觉512→768，音频512→768（可调hidden）
- **Cross-Attention**：音频作为query，视觉作为key/value，融合跨模态信息
- **Temporal Transformer**：4-6层，建模时序依赖
- **对比学习**：InfoNCE损失，增强模态对齐
- **分类头**：时间池化（mean+max）→ MLP → 二分类

### 训练特性
- **混合精度**：自动开启AMP，加速+节省显存
- **梯度裁剪**：max_norm=1.0，防止梯度爆炸
- **Warmup + Cosine**：前3轮线性warmup，后续余弦退火
- **Label Smoothing**：0.1，缓解过拟合
- **数据增强**：随机时序偏移（±3帧）

---

## 预期效果

### 硬件利用
- 4卡A800 并行，总 batch size = 64×4 = 256
- GPU 利用率：80-95%（取决于num_workers和I/O）
- 单 epoch 时间：约 5-10 分钟（取决于样本量）
- 50 epochs 总耗时：约 4-8 小时

### 性能指标
- **训练集**：准确率 > 98%
- **验证集**：准确率 > 90%，AUC > 0.95
- **测试集**：准确率 > 88%，AUC > 0.93

---

## 常见问题

### Q1: OOM (Out of Memory)
**解决**：减小 `--batch_size` 或 `--seq_len` 或 `--hidden`

### Q2: 显存利用率不高
**解决**：增大 `--batch_size`，或增加 `--hidden`/`--num_layers`

### Q3: 训练过慢
**解决**：增大 `--num_workers`（推荐 8-12），确保 `--pin_memory`

### Q4: 验证集准确率低
**解决**：
- 检查数据是否平衡（real/fake 比例）
- 增加 `--epochs` 或减小 `--lr`
- 尝试去掉 `--use_contrastive` 先跑基线

---

## 保存与恢复

训练完成后，最佳模型保存在 `best_model_4gpu_50ep.pt`，包含：
- `model_state_dict`：模型权重
- `optimizer_state_dict`：优化器状态
- `val_acc`、`val_auc`：最佳验证指标
- `config`：训练超参

加载推理：
```python
import torch
from train_crossmodal_ddp import CrossModalTransformer

ckpt = torch.load('best_model_4gpu_50ep.pt')
model = CrossModalTransformer(
    dv=512, da=512, 
    hidden=ckpt['config']['hidden'],
    num_layers=ckpt['config']['num_layers']
)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
```

---

## 下一步

训练完成后可以：
1. 在测试集评估最终指标
2. 实现 `localize_deepfake.py` 做时间段定位
3. 跨数据集泛化测试
4. 加入单模态SOTA做集成

