# LAV-DF 特征提取指南

## 数据集概况（已验证）

- **总视频数**：136,304
  - Real: 36,431
  - Fake: 99,873（Fake:Real ≈ 3:1）
- **Split 分布**：
  - train: 78,703
  - dev: 31,501
  - test: 26,100

## 特征提取流程

### 1. 本地测试（已完成 ✅）

```bash
python test_lavdf_metadata.py
```

验证了：
- 元数据加载正确
- 帧级标签生成逻辑正确（fake_periods 转换为 frame_labels）

### 2. HPC 环境准备

#### 2.1 推送代码到 HPC

```bash
# 在本地
git add prepare_lavdf_features.py test_lavdf_metadata.py docs/LAVDF_FEATURE_EXTRACTION.md
git commit -m "Add LAV-DF feature extraction with frame-level labels"
git push origin main
```

#### 2.2 上传 LAV-DF 数据集到 HPC

```bash
# 在本地，将 LAV-DF 打包（如果尚未上传）
cd D:\LAV-DF
tar -czf LAV-DF.tar.gz LAV-DF/

# 上传到 HPC（示例，使用你的实际方法）
scp LAV-DF.tar.gz xfeng733@hpc:/hpc2ssd/JH_DATA/spooler/xfeng733/

# 在 HPC 解压
ssh xfeng733@hpc
cd /hpc2ssd/JH_DATA/spooler/xfeng733/
tar -xzf LAV-DF.tar.gz
```

#### 2.3 HPC 拉取代码

```bash
cd ~/jhspoolers/voca
git pull origin main
```

### 3. HPC 特征提取

#### 3.1 先测试 1 个视频（快速验证环境）

```bash
source ~/.bashrc
conda activate voca
export PYTHONNOUSERSITE=1

cd ~/jhspoolers/voca

# 测试 1 个视频
python prepare_lavdf_features.py \
  --dataset_root /hpc2ssd/JH_DATA/spooler/xfeng733/LAV-DF \
  --metadata /hpc2ssd/JH_DATA/spooler/xfeng733/LAV-DF/metadata.min.json \
  --output_root /hpc2ssd/JH_DATA/spooler/xfeng733/LAV-DF_feats \
  --use_gpu \
  --splits test \
  --max_videos 1
```

#### 3.2 全量提取（分 split 逐步运行）

**Test Split（优先，用于验证）**：
```bash
python prepare_lavdf_features.py \
  --dataset_root /hpc2ssd/JH_DATA/spooler/xfeng733/LAV-DF \
  --metadata /hpc2ssd/JH_DATA/spooler/xfeng733/LAV-DF/metadata.min.json \
  --output_root /hpc2ssd/JH_DATA/spooler/xfeng733/LAV-DF_feats \
  --use_gpu \
  --splits test \
  --skip_existing
```

**Dev Split**：
```bash
python prepare_lavdf_features.py \
  --dataset_root /hpc2ssd/JH_DATA/spooler/xfeng733/LAV-DF \
  --metadata /hpc2ssd/JH_DATA/spooler/xfeng733/LAV-DF/metadata.min.json \
  --output_root /hpc2ssd/JH_DATA/spooler/xfeng733/LAV-DF_feats \
  --use_gpu \
  --splits dev \
  --skip_existing
```

**Train Split（最大，建议用 GPU 分片并行）**：

选项 1：单 GPU 串行（时间长）
```bash
python prepare_lavdf_features.py \
  --dataset_root /hpc2ssd/JH_DATA/spooler/xfeng733/LAV-DF \
  --metadata /hpc2ssd/JH_DATA/spooler/xfeng733/LAV-DF/metadata.min.json \
  --output_root /hpc2ssd/JH_DATA/spooler/xfeng733/LAV-DF_feats \
  --use_gpu \
  --splits train \
  --skip_existing
```

选项 2：多 GPU 分片并行（推荐）
```bash
# GPU 0：前 1/4
CUDA_VISIBLE_DEVICES=0 python prepare_lavdf_features.py \
  --dataset_root /hpc2ssd/JH_DATA/spooler/xfeng733/LAV-DF \
  --metadata /hpc2ssd/JH_DATA/spooler/xfeng733/LAV-DF/metadata.min.json \
  --output_root /hpc2ssd/JH_DATA/spooler/xfeng733/LAV-DF_feats \
  --use_gpu \
  --splits train \
  --max_videos 19676 \
  --skip_existing &

# GPU 1：第 2/4
CUDA_VISIBLE_DEVICES=1 python prepare_lavdf_features.py \
  --dataset_root /hpc2ssd/JH_DATA/spooler/xfeng733/LAV-DF \
  --metadata /hpc2ssd/JH_DATA/spooler/xfeng733/LAV-DF/metadata.min.json \
  --output_root /hpc2ssd/JH_DATA/spooler/xfeng733/LAV-DF_feats \
  --use_gpu \
  --splits train \
  --max_videos 39352 \
  --skip_existing &

# GPU 2：第 3/4
CUDA_VISIBLE_DEVICES=2 python prepare_lavdf_features.py \
  --dataset_root /hpc2ssd/JH_DATA/spooler/xfeng733/LAV-DF \
  --metadata /hpc2ssd/JH_DATA/spooler/xfeng733/LAV-DF/metadata.min.json \
  --output_root /hpc2ssd/JH_DATA/spooler/xfeng733/LAV-DF_feats \
  --use_gpu \
  --splits train \
  --max_videos 59028 \
  --skip_existing &

# GPU 3：最后 1/4
CUDA_VISIBLE_DEVICES=3 python prepare_lavdf_features.py \
  --dataset_root /hpc2ssd/JH_DATA/spooler/xfeng733/LAV-DF \
  --metadata /hpc2ssd/JH_DATA/spooler/xfeng733/LAV-DF/metadata.min.json \
  --output_root /hpc2ssd/JH_DATA/spooler/xfeng733/LAV-DF_feats \
  --use_gpu \
  --splits train \
  --skip_existing &

wait
```

### 3.3 单卡分片并行（新增 --start_idx / --num_videos）
在一张 A800 上并行跑多个进程，利用率更平滑，`--skip_existing` 避免重复：

```bash
# 训练集前 2 万
CUDA_VISIBLE_DEVICES=0 python prepare_lavdf_features.py \
  --dataset_root /hpc2hdd/home/xfeng733/LAV-DF/LAV-DF \
  --metadata /hpc2hdd/home/xfeng733/LAV-DF/LAV-DF/metadata.min.json \
  --output_root /hpc2hdd/home/xfeng733/LAV-DF_feats \
  --use_gpu \
  --splits train \
  --start_idx 0 --num_videos 20000 \
  --skip_existing &

# 训练集后 2 万
CUDA_VISIBLE_DEVICES=0 python prepare_lavdf_features.py \
  --dataset_root /hpc2hdd/home/xfeng733/LAV-DF/LAV-DF \
  --metadata /hpc2hdd/home/xfeng733/LAV-DF/LAV-DF/metadata.min.json \
  --output_root /hpc2hdd/home/xfeng733/LAV-DF_feats \
  --use_gpu \
  --splits train \
  --start_idx 20000 --num_videos 20000 \
  --skip_existing &

wait
```

如需对半切可提前生成分片 metadata 文件（可选）：
```bash
python - <<'PY'
import json
from pathlib import Path
meta = json.load(open("/hpc2hdd/home/xfeng733/LAV-DF/LAV-DF/metadata.min.json"))
train = [m for m in meta if m["split"] == "train"]
half = len(train)//2
Path("metadata_shards").mkdir(exist_ok=True)
json.dump(train[:half], open("metadata_shards/train_part1.json","w"), ensure_ascii=False)
json.dump(train[half:], open("metadata_shards/train_part2.json","w"), ensure_ascii=False)
PY

# 同一张卡开两个进程（各处理一半）
CUDA_VISIBLE_DEVICES=0 nohup python prepare_lavdf_features.py \
  --dataset_root /hpc2hdd/home/xfeng733/LAV-DF/LAV-DF \
  --metadata metadata_shards/train_part1.json \
  --output_root /hpc2hdd/home/xfeng733/LAV-DF_feats \
  --use_gpu \
  --splits train \
  --skip_existing > train_p1.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python prepare_lavdf_features.py \
  --dataset_root /hpc2hdd/home/xfeng733/LAV-DF/LAV-DF \
  --metadata metadata_shards/train_part2.json \
  --output_root /hpc2hdd/home/xfeng733/LAV-DF_feats \
  --use_gpu \
  --splits train \
  --skip_existing > train_p2.log 2>&1 &

wait
```

### 4. 输出结构

```
LAV-DF_feats/
├── train/
│   ├── 138009/
│   │   ├── bottom_faces/          # 底脸帧图像
│   │   ├── visual_embeddings.npz  # {embeddings, paths, frame_labels}
│   │   ├── audio_embeddings.npz   # {embeddings}
│   │   ├── similarity_stats.npz
│   │   └── similarity_curve.png
│   └── ...
├── dev/
│   └── ...
├── test/
│   └── ...
└── lavdf_summary.csv              # 统计信息
```

**关键**：`visual_embeddings.npz` 现在包含 `frame_labels` 字段：
- `embeddings`: (T, 512) 视觉特征
- `paths`: 提取的帧路径列表
- **`frame_labels`**: (T,) 帧级标签，0=real, 1=fake

### 5. 验证输出

提取完成后，运行验证脚本：

```bash
python -c "
import numpy as np
from pathlib import Path

feat_root = Path('/hpc2ssd/JH_DATA/spooler/xfeng733/LAV-DF_feats')

# 检查一个 fake 样本
fake_npz = feat_root / 'test/000002/visual_embeddings.npz'
data = np.load(fake_npz)
print('Keys:', list(data.keys()))
print('embeddings shape:', data['embeddings'].shape)
print('frame_labels shape:', data['frame_labels'].shape)
print('fake frames:', data['frame_labels'].sum(), '/', len(data['frame_labels']))
print('Expected: 19/116 fake frames')
"
```

## 下一步

特征提取完成后，进行：
1. **分类训练**：使用整体标签（视频级）
2. **定位训练**：使用 `frame_labels`（帧级监督）

定位训练脚本见：`train_lavdf_localization.py`（待创建）


