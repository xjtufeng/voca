# VOCA-Lens Project Status

Audio-Visual Deepfake Detection System based on Fine-portraitist and AniTalker

**Last Updated**: 2024-12-04

---

## Project Overview

**Core Idea**: Detect deepfakes by comparing bottom-face visual features with audio motion features

**Theoretical Basis**:
- Fine-portraitist: Speech-to-bottom-face correlation
- AniTalker: Speech encoder architecture (HuBERT + Conformer)

**Detection Strategy**:
- Real video: High audio-visual consistency
- Fake video: Low audio-visual consistency

---

## Completed Modules

### Phase 1: Visual Pipeline (100% Complete)

#### 1. Bottom-Face Extraction (`face_extractor.py`)
- MediaPipe Face Mesh (468 landmarks)
- Face alignment to 256x256 standard pose
- Bottom-face segmentation using landmark #168 (nose bridge)
- Frame-level processing (sample_rate=1 for all frames)
- Output: aligned faces + bottom-face crops

**Key Features**:
- Adaptive splitting per frame
- Multi-face support (target speaker selection)
- GPU/CPU auto-detection
- Debug visualization mode

**Test Results**:
- test2.mp4: 208/208 frames processed successfully
- Processing speed: ~80ms/frame (CPU)

#### 2. Test Scripts
- `test_face_extraction.py`: Main testing interface
- Successfully validated on test1.mp4 and test2.mp4

---

### Phase 2: Audio Pipeline (100% Complete)

#### 1. Audio Extraction (`audio_extractor.py`)
- Extract 16kHz mono audio from video files
- Using moviepy + librosa
- Output: WAV files ready for speech encoding

**Test Results**:
- test2.mp4: 8.32s audio extracted successfully

#### 2. Speech Motion Encoder (`speech_encoder_anitalker.py`)
- Architecture: HuBERT-large + HAL + Conformer (based on AniTalker)
- Frontend: facebook/hubert-large-ls960-ft (frozen)
- HAL: Hierarchical Aggregation Layer (25 learnable weights)
- Downsample: Conv1D 50Hz → 25Hz
- Backend: 4-layer Conformer encoder
- Output: (T, 512) motion latent @ 25fps

**Components from AniTalker**:
- Conformer encoder structure (100% identical)
- HAL weighting mechanism (100% identical)
- Downsampling strategy (100% identical)

**Test Results**:
- test2_audio.wav: (208, 512) motion latent extracted
- 208 frames @ 25fps = 8.32s (perfectly aligned with video)
- Mean ≈ 0, Std ≈ 1 (normalized features)

---

### Phase 3: Visual Encoding (In Progress)

#### 1. Face Encoder Selection
- ~~VGG-Face (.t7 weights): ABANDONED (loading issues)~~
- ~~FaceNet (facenet-pytorch): Temporary solution~~
- **InsightFace ArcFace: CURRENT SOLUTION**

#### 2. InsightFace Integration (`face_encoder_insightface.py`)
- Model: buffalo_l (ArcFace ResNet-based)
- Pretrained on face recognition datasets
- Output: (T, 512) face embeddings

**Features**:
- Automatic padding for bottom-face images (preserves aspect ratio)
- Batch processing support
- L2-normalized embeddings

**Status**: Code ready, awaiting final testing

---

## Current Architecture

```
Input Video (e.g., test2.mp4, 8.32s, 25fps)
    |
    +------ Video Track ------+         +------ Audio Track ------+
    |                         |         |                         |
    v                         |         v                         |
MediaPipe Face Mesh          |    moviepy Audio Extract         |
    |                         |         |                         |
    v                         |         v                         |
Bottom-face Extraction       |    16kHz Mono WAV                 |
(208 frames, 256x156)        |    (8.32s, 133k samples)          |
    |                         |         |                         |
    v                         |         v                         |
Padding to square            |    HuBERT-large (frozen)          |
(256x256, black borders)     |         |                         |
    |                         |         v                         |
    v                         |    HAL (25 layers)                |
InsightFace ArcFace          |         |                         |
(buffalo_l model)            |         v                         |
    |                         |    Conv1D (50→25Hz)               |
    v                         |         |                         |
z_visual (208, 512)          |         v                         |
                             |    4-layer Conformer              |
                             |         |                         |
                             |         v                         |
                             |    z_audio (208, 512)             |
                             |                                   |
    +------------------------+-----------------+                 |
                             v                                   |
                Frame-level Alignment                            |
                             |                                   |
                             v                                   |
                Cosine Similarity (208,)                         |
                             |                                   |
                             v                                   |
                Statistical Features                             |
                             |                                   |
                             v                                   |
                Lightweight Classifier                           |
                             |                                   |
                             v                                   |
                    Real / Fake Decision                         |
```

---

## Technical Stack

| Module | Technology | Status |
|--------|-----------|--------|
| Face Detection | MediaPipe Face Mesh | ✅ Stable |
| Face Alignment | Affine Transform (eye-based) | ✅ Stable |
| Bottom-face Crop | Landmark #168 adaptive split | ✅ Stable |
| Visual Encoder | InsightFace ArcFace (buffalo_l) | 🔄 Testing |
| Audio Extraction | moviepy + librosa | ✅ Stable |
| Speech Encoder | HuBERT + HAL + Conformer | ✅ Tested |
| Feature Alignment | Frame-level timestamp matching | ⏳ Pending |
| Similarity Metric | Cosine Similarity | ⏳ Pending |
| Classifier | Logistic Regression / MLP | ⏳ Pending |

---

## Environment Setup

### Environment 1: Base (Python 3.12)
**Purpose**: Audio processing, HuBERT encoder, general testing

**Installation**:
```bash
pip install -r requirements_base.txt
```

**Key Dependencies**:
- torch, torchvision
- transformers (HuBERT)
- espnet (Conformer)
- librosa, soundfile, moviepy
- opencv-python, mediapipe

### Environment 2: voca-insight (Python 3.10, Conda)
**Purpose**: InsightFace face encoding

**Installation**:
```bash
conda create -n voca-insight python=3.10
conda activate voca-insight
pip install -r requirements_insightface.txt
```

**Key Dependencies**:
- torch, torchvision
- insightface
- onnxruntime
- opencv-python

---

## Performance Metrics

### test2.mp4 Processing Results

| Module | Input | Output | Time | Status |
|--------|-------|--------|------|--------|
| Audio Extraction | test2.mp4 | test2_audio.wav (16kHz, 8.32s) | ~2s | ✅ |
| Speech Encoding | test2_audio.wav | (208, 512) motion latent | ~2s | ✅ |
| Face Extraction | test2.mp4 | 208 bottom-face images | ~16s | ✅ |
| Face Encoding | 208 images | (208, 512) embeddings | TBD | 🔄 |

**Frame Alignment**: Perfect (208 audio frames ↔ 208 video frames @ 25fps)

---

## Training Requirements

### Current Stage (Zero-shot Baseline)
**What needs training**: Only lightweight classifier
- Input: Statistical features from similarity curve (5-10 dims)
- Output: Real / Fake binary classification
- Model: Logistic Regression or 2-layer MLP
- Data needed: 20-50 labeled videos

**What does NOT need training**:
- ✅ HuBERT: Frozen (pretrained)
- ✅ HAL + Conformer: Can use random init (or load AniTalker weights)
- ✅ InsightFace ArcFace: Frozen (pretrained)

### Future Stage (If baseline performance insufficient)
**Optional training targets**:
- 🔄 HAL + Conv1D + Conformer (~5M params): Align audio latent to bottom-face motion
- 🔄 Visual projection head (~500K params): Adapt ArcFace to bottom-face distribution

**Training approach**: 
- L2 + Cosine loss between z_audio and z_visual on real videos
- Or contrastive learning (InfoNCE)

---

## Next Steps

### Immediate (This Week)
1. ✅ Complete InsightFace environment setup
2. 🔄 Test `face_encoder_insightface.py` on test2 bottom-faces
3. ⏳ Create `compare_audio_visual.py` for similarity computation
4. ⏳ Visualize similarity curves

### Short-term (1-2 Weeks)
1. Extract features for multiple videos (real + fake)
2. Implement statistical feature extraction
3. Train lightweight classifier
4. Evaluate baseline performance (AUC, Accuracy)

### Medium-term (1 Month)
1. Ablation studies:
   - Full-face vs bottom-face
   - Fixed split vs adaptive split
   - Different similarity metrics
2. Fine-tune visual encoder (projection head)
3. Optimize for ICML submission

---

## Known Issues & Solutions

1. **InsightFace requires MSVC compiler on Windows**
   - Solution: Install Microsoft C++ Build Tools
   - Alternative: Use conda environment with Python 3.10

2. **Dual Miniconda installations causing conflicts**
   - Solution: Keep only one (C:\ProgramData\miniconda3)
   - Clean user directory installation

3. **Bottom-face aspect ratio mismatch (156x256 vs 256x256)**
   - Solution: Padding to square before feeding to ArcFace
   - Preserves aspect ratio, no distortion

---

## File Organization

### Core Modules
- `face_extractor.py`: Bottom-face extraction (MediaPipe)
- `face_encoder_insightface.py`: Visual encoding (InsightFace ArcFace)
- `speech_encoder_anitalker.py`: Audio encoding (HuBERT + Conformer)
- `audio_extractor.py`: Audio extraction from video

### Test Scripts
- `test_face_extraction.py`: Test bottom-face extraction
- `verify_features.py`: Verify feature quality

### Documentation
- `PROJECT_STATUS.md`: This file
- `PROJECT_STRUCTURE.md`: Detailed architecture
- `CLEANUP_SUMMARY.md`: Historical cleanup notes
- `AniTalker.pdf`: Reference paper

### Data
- `test2.mp4`: Main test video (8.32s, 25fps, 208 frames)
- `test2_audio.wav`: Extracted audio
- `test_video_output/`: Processing results
  - `aligned_faces/`: 208 full-face images (256x256)
  - `bottom_faces/`: 208 bottom-face images (~156x256)

### Models
- `models/`: Downloaded model files
- External: `external/AniTalker/`: Reference implementation

---

## Citation & References

When using this project, please cite:

**AniTalker** (Speech Encoder Architecture):
```bibtex
@inproceedings{liu2024anitalker,
  title={AniTalker: Animate Vivid and Diverse Talking Faces through Identity-Decoupled Facial Motion Encoding},
  author={Liu, Tao and others},
  booktitle={ACM MM},
  year={2024}
}
```

**InsightFace** (Face Recognition):
```bibtex
@inproceedings{deng2019arcface,
  title={ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
  author={Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle={CVPR},
  year={2019}
}
```

**Fine-portraitist** (Theoretical Basis):
- Reference paper provided in project directory

---

**Project Status**: Active Development  
**Target Venue**: ICML 2025  
**Current Phase**: Baseline Implementation & Testing
