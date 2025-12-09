# VOCA-Lens 项目结构说明

## 📁 目录结构

```
VOCA-Lens/
│
├── 📄 README.md                    # 项目主文档
├── 📄 PROJECT_STATUS.md            # 项目进度跟踪
├── 📄 PROJECT_STRUCTURE.md         # 本文件：项目结构说明
│
├── 🔧 核心代码模块
│   ├── face_extractor.py          # 底脸提取器（MediaPipe）
│   ├── vgg_encoder.py             # VGG-Face 编码器
│   ├── vgg_face_decoder.py        # VGG-Face 解码器（含CBAM）
│   └── video_process.py            # 音频处理（待重构为audio_processor.py）
│
├── 🧪 测试脚本
│   └── test_face_extraction.py    # 底脸提取功能测试
│
├── 📚 文档与参考
│   ├── Fine_portraitist_CameraReady(1).pdf  # 参考论文
│   └── README_bottom_face.md      # 底脸提取详细说明（旧文档）
│
├── 🗂️ 模型文件
│   ├── VGG_FACE.t7                # VGG-Face 预训练权重
│   └── shape_predictor_68_face_landmarks.dat  # dlib模型（已不使用，保留）
│
├── 📦 测试数据
│   ├── test1.mp4                  # 测试视频 1 (125帧, 5.0s)
│   └── test2.mp4                  # 测试视频 2 (208帧, 8.32s)
│
├── 📤 输出结果
│   ├── test_video_output/         # test1.mp4 处理结果
│   └── test2_output/              # test2.mp4 处理结果
│
└── 🗑️ 旧代码（保留参考）
    └── legacy_face_extractor.py   # 原始人脸提取代码（dlib版本）
```

## 📝 文件说明

### 核心模块

#### `face_extractor.py`
**功能**: 使用 MediaPipe 进行底脸提取
- 人脸检测（MediaPipe Face Mesh）
- 人脸对齐（256×256 标准姿态）
- 底脸分割（基于 landmark 168 - 鼻梁中点）

**主要类**: `BottomFaceExtractor`

**关键方法**:
- `process_image()` - 处理单张图像
- `process_video_frames()` - 批量处理视频帧
- `extract_bottom_face()` - 提取底脸（使用 MediaPipe landmark 168）

#### `vgg_encoder.py`
**功能**: VGG-Face 编码器
- 加载预训练权重（VGG_FACE.t7）
- 提取人脸特征（4096维 fc7 特征）

**主要类**: `VGG_Model`

#### `vgg_face_decoder.py`
**功能**: VGG-Face 解码器（用于生成任务）
- VGG_16: 编码器
- CBAM_Module: 注意力机制
- Face_Decoder: 解码器

**主要类**: `VGG_16`, `CBAM_Module`, `Face_Decoder`

#### `video_process.py`
**功能**: 视频和音频处理
- 视频剪辑
- 音频提取
- 音频重采样（16kHz）

**状态**: 待重构为 `audio_processor.py`

### 测试脚本

#### `test_face_extraction.py`
**功能**: 测试底脸提取功能

**用法**:
```bash
# 测试视频
python test_face_extraction.py video <video_path> [sample_rate]

# 测试图像
python test_face_extraction.py image <image_path>
```

### 旧代码

#### `legacy_face_extractor.py`
**功能**: 原始的人脸提取代码（使用 dlib）
- 保留作为参考
- 已不再使用（已替换为 MediaPipe 版本）

## 🔄 文件重命名历史

| 旧名称 | 新名称 | 原因 |
|--------|--------|------|
| `Encoder(1).py` | `vgg_encoder.py` | 规范化命名 |
| `face_decoder.py` | `vgg_face_decoder.py` | 明确模块用途 |
| `bottom_face_extractor.py` | `face_extractor.py` | 简化名称 |
| `test_bottom_face.py` | `test_face_extraction.py` | 更清晰的测试命名 |
| `face_img.py` | `legacy_face_extractor.py` | 标记为旧代码 |

## 🗑️ 已删除的文件

以下文件已删除（不再需要）：
- `debug_video.py` - 调试脚本
- `debug_video2.py` - 调试脚本
- `test_video2.py` - 临时测试脚本
- `install_miniconda_dlib.md` - 安装指南（已完成）
- `SUCCESS_REPORT.md` - 临时报告
- `shape_predictor_68_face_landmarks.dat.bz2` - 压缩包（已解压）

## 📊 输出目录结构

处理视频后，会在输出目录生成：

```
<output_dir>/
├── aligned_faces/      # 对齐的完整人脸 (256×256 PNG)
│   └── frame_XXXXXX.png
│
├── bottom_faces/       # 底脸区域 (H×256 PNG, H≈156)
│   └── frame_XXXXXX.png
│
├── debug/              # 调试图像（带分割线标注）
│   └── frame_XXXXXX.png
│
└── statistics.json     # 处理统计信息
```

## 🔧 依赖关系

```
face_extractor.py
  ├── mediapipe
  ├── opencv-python
  └── numpy

vgg_encoder.py
  ├── torch
  └── torchfile

vgg_face_decoder.py
  ├── torch
  └── torchfile

video_process.py
  ├── opencv-python
  ├── moviepy
  └── librosa
```

## 📈 下一步计划

### 即将创建的文件

1. **`audio_processor.py`** - 重构 `video_process.py`
   - 音频提取
   - 帧-音频对齐
   - Wav2Vec 2.0 集成

2. **`feature_extractor.py`** - 特征提取封装
   - 整合 VGG Encoder
   - 批量处理底脸图像
   - 保存 embeddings

3. **`deepfake_detector.py`** - 主检测器
   - 相似度计算
   - 异常检测
   - 时间定位

4. **`visualizer.py`** - 可视化工具
   - 时序曲线
   - 热图生成
   - 报告输出

## 📝 命名规范

### Python 文件
- 使用小写字母和下划线：`face_extractor.py`
- 模块名应清晰描述功能
- 避免使用数字和特殊字符

### 类名
- 使用大驼峰：`BottomFaceExtractor`
- 清晰描述类的用途

### 函数名
- 使用小写字母和下划线：`process_image()`
- 动词开头，描述动作

### 变量名
- 使用小写字母和下划线：`split_y`
- 清晰描述变量含义

