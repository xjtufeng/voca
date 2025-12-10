# HPC 快速使用指南（FakeAVCeleb v1.2，CPU 流程）

## 0. 路径约定
- 工作目录：`/hpc2hdd/home/xfeng733/jhspoolers`
- 原始压缩包：`FakeAVCeleb_v1.2.zip`
- 解压后实际内容在：`FakeAVCeleb_v1.2/FakeAVCeleb_v1.2/`
  - 子目录：`RealVideo-RealAudio`（真）、`RealVideo-FakeAudio`、`FakeVideo-RealAudio`、`FakeVideo-FakeAudio`（均视为假）
- 兼容数据根目录（将要创建软链接）：`FakeAV_stage/`
- 特征输出目录：`FakeAV_feats/`

## 1. 解压（已完成示例）
```bash
cd /hpc2hdd/home/xfeng733/jhspoolers
python -m zipfile -e FakeAVCeleb_v1.2.zip FakeAVCeleb_v1.2
```

## 2. 构建兼容的 real/fake 目录（保留层级，递归搜索，无需扁平化）
```bash
cd /hpc2hdd/home/xfeng733/jhspoolers

python scripts/create_fakeav_stage.py \
  --src_root FakeAVCeleb_v1.2/FakeAVCeleb_v1.2 \
  --stage_root /hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_stage \
  --force

# 快速检查
find /hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_stage/real -type l | head
find /hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_stage/fake -type l | head
find /hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_stage/real -type l | wc -l
find /hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_stage/fake -type l | wc -l
```

`dataset_root` 示例：`/hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_stage`

## 3. 环境准备（CPU 跑通）
```bash
cd /hpc2hdd/home/xfeng733/jhspoolers
# 如有 conda，可：
# conda create -n voca python=3.10 -y
# conda activate voca

pip install -r requirements_base.txt
pip install -r requirements_insightface.txt  # 如需跑 InsightFace
pip install scikit-learn matplotlib
```

## 4. 特征提取（保持 encoder 不变，CPU）
```bash
python prepare_features_dataset.py \
  --dataset_root /hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_stage \
  --output_root /hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_feats \
  --use_gpu    # 若有可用 GPU
```
输出将位于 `FakeAV_feats/{real,fake}/*/visual_embeddings.npz` 和 `audio_embeddings.npz`。

## 5. 训练跨模态轻量基线（CPU）
```bash
python train_crossmodal_baseline.py \
  --features_root /hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_feats \
  --device cuda \  # 有 GPU 时
  --batch_size 8 \
  --epochs 10 \
  --seq_len 160 \
  --lr 5e-4 \
  --save_path crossmodal_baseline.pt
```
目标：验证/测试准确率 ≥ 70%（如不足，可增大 epochs 或减小 lr）。

## 6. 常见问题
- 文件重名导致软链失败：已在脚本中保留原子目录层级，避免重名。
- CPU 速度慢：可先小样本验证流程，或切换 FaceNet 编码器（较轻）。
- 缺依赖：按第 3 步安装；InsightFace 需 onnxruntime 等，已在 `requirements_insightface.txt`。


