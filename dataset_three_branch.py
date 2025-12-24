"""
Dataset for three-branch joint training
Supports both pre-extracted features and raw video frame loading
"""
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional
import hashlib
import cv2

_VIDEO_INDEX_CACHE: Dict[str, Dict[str, Path]] = {}


def _slugify_path(path: Path) -> str:
    """Match prepare_features_dataset.py: join path parts (without suffix) with underscores."""
    return "_".join(path.with_suffix("").parts)


def _split_bucket(video_id: str) -> int:
    """Stable 0..9 bucket for train/dev/test splitting."""
    return int(hashlib.md5(video_id.encode("utf-8")).hexdigest(), 16) % 10


class ThreeBranchDataset(Dataset):
    """
    Dataset for three-branch training
    Loads audio features, visual features, and optionally raw video frames
    """
    def __init__(
        self,
        features_root: str,
        video_root: Optional[str] = None,
        split: str = 'train',
        max_frames: int = 256,
        load_video_frames: bool = False,
        target_fps: int = 25,
        ignore_missing_videos: bool = True
    ):
        """
        Args:
            features_root: Path to extracted features (audio + InsightFace visual)
            video_root: Path to original videos (for CLIP frame loading)
            split: Dataset split (train/dev/test)
            max_frames: Maximum frames per video
            load_video_frames: Whether to load raw video frames
            target_fps: Target FPS for frame extraction
        """
        self.features_root = Path(features_root)
        self.video_root = Path(video_root) if video_root else None
        self.split = split
        self.max_frames = max_frames
        self.load_video_frames = load_video_frames
        self.target_fps = target_fps
        self.ignore_missing_videos = ignore_missing_videos
        
        # Build video index if loading frames
        self.video_index = {}
        if self.load_video_frames and self.video_root:
            self._build_video_index()
        
        # Collect samples
        self.samples = []
        self._load_samples()
        
        # Compute class weights
        self._compute_class_weights()
        
        print(f"[INFO] {split} dataset: {len(self.samples)} samples")
    
    def _build_video_index(self):
        """Build video_id -> video_path index for fast lookup"""
        cache_key = str(self.video_root)
        if cache_key in _VIDEO_INDEX_CACHE:
            self.video_index = _VIDEO_INDEX_CACHE[cache_key]
            print(f"[INFO] Reusing cached video index: {len(self.video_index)} entries")
            return

        print(f"[INFO] Building video index from {self.video_root}...")
        
        if not self.video_root.exists():
            print(f"[WARN] Video root does not exist: {self.video_root}")
            return
        
        # Recursively find all .mp4 files
        all_videos = list(self.video_root.rglob('*.mp4'))
        print(f"[INFO] Found {len(all_videos)} video files")
        
        for video_path in all_videos:
            # Prefer a key compatible with prepare_features_dataset.py
            # video_id = slugify(relpath_under_label_dir)
            vid_key = None
            try:
                if "real" in video_path.parts:
                    rel = video_path.relative_to(self.video_root / "real")
                    vid_key = _slugify_path(rel)
                elif "fake" in video_path.parts:
                    rel = video_path.relative_to(self.video_root / "fake")
                    vid_key = _slugify_path(rel)
            except Exception:
                vid_key = None

            # Also keep basename (stem) as fallback
            keys = []
            if vid_key:
                keys.append(vid_key)
            keys.append(video_path.stem)

            for k in keys:
                if k not in self.video_index:
                    self.video_index[k] = video_path
        
        print(f"[INFO] Video index built: {len(self.video_index)} unique video IDs")
        _VIDEO_INDEX_CACHE[cache_key] = self.video_index

    def _is_in_split(self, video_id: str) -> bool:
        b = _split_bucket(video_id)
        if self.split == "train":
            return b <= 7  # 80%
        if self.split == "dev":
            return b == 8  # 10%
        if self.split == "test":
            return b == 9  # 10%
        return True
    
    def _load_samples(self):
        """Load all samples from features directory"""
        split_dirs = {
            'train': ['real', 'fake'],
            'dev': ['real', 'fake'],
            'test': ['real', 'fake']
        }
        
        if self.split not in split_dirs:
            raise ValueError(f"Unknown split: {self.split}")
        
        for label_dir in split_dirs[self.split]:
            label = 1 if label_dir == 'fake' else 0
            label_path = self.features_root / label_dir
            
            if not label_path.exists():
                print(f"[WARN] Directory not found: {label_path}")
                continue
            
            # Each video is a directory: label_path/<video_id>/{audio_embeddings.npz, visual_embeddings.npz, ...}
            for video_dir in sorted([p for p in label_path.iterdir() if p.is_dir()]):
                video_id = video_dir.name
                if not self._is_in_split(video_id):
                    continue

                audio_npz = video_dir / "audio_embeddings.npz"
                visual_npz = video_dir / "visual_embeddings.npz"
                if (not audio_npz.exists()) or (not visual_npz.exists()):
                    continue

                video_path = None
                if self.load_video_frames and self.video_root:
                    video_path = self.video_index.get(video_id)
                    if video_path is None and not self.ignore_missing_videos:
                        continue

                self.samples.append({
                    'audio_path': audio_npz,
                    'visual_path': visual_npz,
                    'video_path': video_path,
                    'label': label,
                    'video_id': video_id
                })
    
    def _compute_class_weights(self):
        """Compute pos_weight for BCE loss"""
        num_real = sum(1 for s in self.samples if s['label'] == 0)
        num_fake = sum(1 for s in self.samples if s['label'] == 1)
        
        if num_fake > 0:
            self.pos_weight = num_real / num_fake
        else:
            self.pos_weight = 1.0
        
        print(f"[INFO] {self.split}: real={num_real}, fake={num_fake}, pos_weight={self.pos_weight:.3f}")
    
    def __len__(self):
        return len(self.samples)
    
    def _load_video_frames(self, video_path: Path) -> Optional[np.ndarray]:
        """
        Load video frames
        Returns: [T, H, W, C] numpy array (RGB, uint8)
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25
        
        # Sample frames at target_fps
        frame_interval = max(1, int(fps / self.target_fps))
        
        frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
                if len(frames) >= self.max_frames:
                    break
            
            frame_idx += 1
        
        cap.release()
        
        if not frames:
            return None
        
        return np.array(frames)  # [T, H, W, C]
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load features (stored in separate npz files, key is 'embeddings')
        audio_data = np.load(sample['audio_path'], allow_pickle=True)
        visual_data = np.load(sample['visual_path'], allow_pickle=True)

        audio_feats = audio_data['embeddings']  # [T, a_dim]
        if len(audio_feats) > self.max_frames:
            audio_feats = audio_feats[:self.max_frames]
        
        visual_feats = visual_data['embeddings']  # [T, v_dim]
        if len(visual_feats) > self.max_frames:
            visual_feats = visual_feats[:self.max_frames]
        
        # Ensure same length
        min_len = min(len(audio_feats), len(visual_feats))
        audio_feats = audio_feats[:min_len]
        visual_feats = visual_feats[:min_len]
        
        # Load video frames if needed
        video_frames = None
        if self.load_video_frames and sample['video_path']:
            frames_np = self._load_video_frames(sample['video_path'])
            if frames_np is not None:
                # Truncate to match features
                frames_np = frames_np[:min_len]
                # Convert to tensor [T, C, H, W] and normalize to [0, 1]
                video_frames = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0
        
        # Convert to tensors
        audio_feats = torch.from_numpy(audio_feats).float()
        visual_feats = torch.from_numpy(visual_feats).float()
        label = torch.tensor(sample['label'], dtype=torch.float32)
        
        result = {
            'audio': audio_feats,
            'visual': visual_feats,
            'label': label,
            'video_id': sample['video_id']
        }
        
        if video_frames is not None:
            result['video_frames'] = video_frames
        
        return result


def collate_three_branch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for variable-length sequences
    Pads all sequences to max length in batch
    """
    # Find max length
    max_len = max(b['audio'].size(0) for b in batch)
    
    batch_size = len(batch)
    audio_dim = batch[0]['audio'].size(-1)
    visual_dim = batch[0]['visual'].size(-1)
    
    # Initialize tensors
    audio_batch = torch.zeros(batch_size, max_len, audio_dim)
    visual_batch = torch.zeros(batch_size, max_len, visual_dim)
    labels = torch.zeros(batch_size)
    mask = torch.ones(batch_size, max_len, dtype=torch.bool)  # True = padded
    
    # Handle video frames (some samples may fall back to feature-only)
    has_video_frames = any('video_frames' in b for b in batch)
    if has_video_frames:
        # Get dimensions from first sample
        first = next(b for b in batch if 'video_frames' in b)
        _, C, H, W = first['video_frames'].shape
        video_frames_batch = torch.zeros(batch_size, max_len, C, H, W)
    
    video_ids = []
    
    for i, sample in enumerate(batch):
        seq_len = sample['audio'].size(0)
        
        audio_batch[i, :seq_len] = sample['audio']
        visual_batch[i, :seq_len] = sample['visual']
        labels[i] = sample['label']
        mask[i, :seq_len] = False  # False = valid
        video_ids.append(sample['video_id'])
        
        if has_video_frames and 'video_frames' in sample:
            video_frames_batch[i, :seq_len] = sample['video_frames']
    
    result = {
        'audio': audio_batch,
        'visual': visual_batch,
        'label': labels,
        'mask': mask,
        'video_id': video_ids
    }
    
    if has_video_frames:
        result['video_frames'] = video_frames_batch
    
    return result


def get_three_branch_dataloaders(
    features_root: str,
    video_root: Optional[str] = None,
    splits: List[str] = ['train', 'dev'],
    batch_size: int = 8,
    num_workers: int = 4,
    max_frames: int = 256,
    load_video_frames: bool = False,
    target_fps: int = 25,
    ignore_missing_videos: bool = True
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for three-branch training
    
    Args:
        features_root: Path to extracted features
        video_root: Path to original videos (for CLIP)
        splits: List of splits to load
        batch_size: Batch size
        num_workers: Number of worker processes
        max_frames: Max frames per video
        load_video_frames: Whether to load raw video frames
        target_fps: Target FPS for frame extraction
    
    Returns:
        Dictionary of {split: DataLoader}
    """
    dataloaders = {}
    
    for split in splits:
        dataset = ThreeBranchDataset(
            features_root=features_root,
            video_root=video_root,
            split=split,
            max_frames=max_frames,
            load_video_frames=load_video_frames,
            target_fps=target_fps,
            ignore_missing_videos=ignore_missing_videos
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            collate_fn=collate_three_branch,
            pin_memory=True,
            drop_last=(split == 'train')
        )
        
        dataloaders[split] = dataloader
    
    # Return pos_weight from train dataset
    if 'train' in splits:
        train_dataset = dataloaders['train'].dataset
        dataloaders['pos_weight'] = train_dataset.pos_weight
    
    return dataloaders

