#!/usr/bin/env python3
"""
LAV-DF Localization Dataset
Loads video features with frame-level labels for temporal deepfake localization
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
from typing import List, Dict, Tuple
import json


class LAVDFLocalizationDataset(Dataset):
    """
    Dataset for frame-level deepfake localization
    Loads visual/audio embeddings + frame_labels from extracted features
    """
    def __init__(
        self, 
        features_root: str,
        split: str = 'train',
        max_frames: int = 512,
        stride: int = 1,
        min_frames: int = 10,
        event_centric_prob: float = 0.5
    ):
        """
        Args:
            features_root: Path to extracted features root
            split: 'train', 'dev', or 'test'
            max_frames: Maximum frames per video (truncate if longer)
            stride: Frame sampling stride (1=all frames, 2=every other frame)
            min_frames: Minimum frames required (skip videos with fewer frames)
            event_centric_prob: For fake videos in train split, probability of event-centric sampling
                                0.5 = balanced (recommended), 0.7 = more focus on events, 1.0 = always event-centric
        """
        self.features_root = Path(features_root)
        self.split = split
        self.max_frames = max_frames
        self.stride = stride
        self.min_frames = min_frames
        self.event_centric_prob = event_centric_prob
        
        # Scan for video feature directories
        self.samples = self._scan_samples()
        
        # Compute class statistics
        self._compute_class_stats()
        
        print(f"[INFO] {split} split: {len(self.samples)} videos")
        print(f"[INFO] Total frames: {self.total_frames}, Fake frames: {self.fake_frames} ({self.fake_ratio*100:.2f}%)")
    
    def _scan_samples(self) -> List[Dict]:
        """Scan feature directories and collect valid samples"""
        split_dir = self.features_root / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        # If loading train split, check for dev_temp and exclude those samples
        dev_temp_ids = set()
        if self.split == 'train':
            dev_temp_dir = self.features_root / 'dev_temp'
            if dev_temp_dir.exists():
                dev_temp_ids = {p.name for p in dev_temp_dir.iterdir() if p.is_symlink() or p.is_dir()}
                if dev_temp_ids:
                    print(f"[INFO] Found dev_temp with {len(dev_temp_ids)} samples, excluding from train")
        
        samples = []
        # Sort for determinism across ranks (important for DDP reproducibility)
        for video_dir in sorted(split_dir.iterdir(), key=lambda p: p.name):
            if not video_dir.is_dir():
                continue
            
            # Skip if this sample is in dev_temp (for train split)
            if video_dir.name in dev_temp_ids:
                continue
            
            visual_file = video_dir / "visual_embeddings.npz"
            audio_file = video_dir / "audio_embeddings.npz"
            
            if not (visual_file.exists() and audio_file.exists()):
                continue
            
            # Quick check: validate BOTH npz files are readable and consistent enough to train.
            # This prevents mid-epoch DataLoader crashes (which lead to DDP/NCCL hangs).
            try:
                visual_data = np.load(visual_file)
                if 'embeddings' not in visual_data.files:
                    print(f"[WARN] Missing embeddings in {visual_file}, skipping")
                    continue
                
                # Actually read visual embeddings to trigger CRC check (not just check key exists)
                _ = visual_data['embeddings']
                num_frames = len(visual_data['embeddings'])
                if num_frames < self.min_frames:
                    continue
                
                # Check if frame_labels exist
                if 'frame_labels' not in visual_data.files:
                    print(f"[WARN] Missing frame_labels in {visual_file}, skipping")
                    continue

                # Validate audio file can be opened and has embeddings.
                # Actually read to ensure CRC is valid (not just check key exists).
                audio_data = np.load(audio_file)
                if 'embeddings' not in audio_data.files:
                    print(f"[WARN] Missing embeddings in {audio_file}, skipping")
                    continue
                
                # Actually read audio embeddings to trigger CRC check
                _ = audio_data['embeddings']
                
                samples.append({
                    'video_id': video_dir.name,
                    'visual_file': visual_file,
                    'audio_file': audio_file,
                    'num_frames': num_frames
                })
            except Exception as e:
                print(f"[WARN] Failed to load {visual_file}: {e}")
                continue
        
        return samples
    
    def _compute_class_stats(self):
        """Compute frame-level class statistics for pos_weight"""
        total_frames = 0
        fake_frames = 0
        
        for sample in self.samples:
            try:
                data = np.load(sample['visual_file'])
                frame_labels = data['frame_labels']
                total_frames += len(frame_labels)
                fake_frames += frame_labels.sum()
            except:
                continue
        
        self.total_frames = total_frames
        self.fake_frames = int(fake_frames)
        self.real_frames = total_frames - self.fake_frames
        self.fake_ratio = self.fake_frames / max(total_frames, 1)
        
        # Compute pos_weight for BCEWithLogitsLoss
        if self.fake_frames > 0:
            self.pos_weight = self.real_frames / self.fake_frames
        else:
            self.pos_weight = 1.0
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _get_event_centric_start(self, frame_labels: np.ndarray, total_frames: int) -> int:
        """
        Compute event-centric sampling start position.
        Centers the window on a fake segment to ensure boundary learning.
        
        Args:
            frame_labels: [T] frame labels
            total_frames: Total number of frames before truncation
        
        Returns:
            start: Starting index for window
        """
        # Find all fake segments
        fake_indices = np.where(frame_labels == 1)[0]
        
        if len(fake_indices) == 0:
            # Fallback to random (should not happen if called only for has_fake_frames)
            return np.random.randint(0, total_frames - self.max_frames + 1)
        
        # Find segment boundaries
        segments = []
        start_seg = fake_indices[0]
        prev_idx = fake_indices[0]
        
        for idx in fake_indices[1:]:
            if idx - prev_idx > 1:  # Gap detected, new segment
                segments.append((start_seg, prev_idx))
                start_seg = idx
            prev_idx = idx
        segments.append((start_seg, prev_idx))  # Last segment
        
        # Randomly select one segment
        seg_start, seg_end = segments[np.random.randint(len(segments))]
        
        # Center window on segment
        seg_center = (seg_start + seg_end) // 2
        window_start = seg_center - self.max_frames // 2
        
        # Ensure window is within bounds
        window_start = max(0, min(window_start, total_frames - self.max_frames))
        
        return window_start
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        try:
            # Load visual embeddings + frame_labels
            visual_data = np.load(sample['visual_file'])
            visual_emb = visual_data['embeddings']  # [T_v, 512]
            frame_labels = visual_data['frame_labels']  # [T_v]
            frame_ids = visual_data['frame_ids'] if 'frame_ids' in visual_data.files else None
            fps = float(visual_data['fps']) if 'fps' in visual_data.files else None
            
            # Load audio embeddings
            audio_data = np.load(sample['audio_file'])
            audio_emb = audio_data['embeddings']  # [T_a, 1024]
        except Exception as e:
            # Safety net: if loading fails (corrupted file, IO error, etc.),
            # skip this sample and try the next one to avoid DDP hangs
            print(f"[ERROR] Failed to load {sample['video_id']}: {e}, trying next sample")
            # Return next sample (wrap around if at end)
            next_idx = (idx + 1) % len(self.samples)
            return self.__getitem__(next_idx)
        
        # Sample frames with stride
        if self.stride > 1:
            indices = np.arange(0, len(visual_emb), self.stride)
            visual_emb = visual_emb[indices]
            frame_labels = frame_labels[indices]
            if frame_ids is not None:
                frame_ids = frame_ids[indices]
        
        # Truncate if too long
        orig_T_v = len(visual_emb)
        start = 0
        end = orig_T_v
        frame_ids_full = frame_ids
        if len(visual_emb) > self.max_frames:
            # Video-level label (check if any fake frame)
            has_fake_frames = frame_labels.sum() > 0
            
            if self.split == 'train' and has_fake_frames and np.random.rand() < self.event_centric_prob:
                # Event-centric sampling: center on a fake segment
                start = self._get_event_centric_start(frame_labels, len(visual_emb))
            elif self.split == 'train':
                # Random crop
                start = np.random.randint(0, len(visual_emb) - self.max_frames + 1)
            else:
                # Val/test: center crop
                start = (len(visual_emb) - self.max_frames) // 2
            
            end = start + self.max_frames
            visual_emb = visual_emb[start:end]
            frame_labels = frame_labels[start:end]
            if frame_ids is not None:
                frame_ids = frame_ids[start:end]
        
        # Align audio to visual frames (crop audio window, then interpolate)
        T_v = len(visual_emb)
        T_a = len(audio_emb)
        
        if T_a != T_v:
            audio_seg = audio_emb
            if frame_ids is not None and fps is not None and len(frame_ids_full) > 0:
                t0 = float(frame_ids_full[0]) / fps
                t1 = float(frame_ids_full[-1]) / fps
                total_duration = t1 - t0
                if total_duration > 0 and T_a > 1 and frame_ids is not None and len(frame_ids) > 0:
                    t_start = float(frame_ids[0]) / fps
                    t_end = float(frame_ids[-1]) / fps
                    a_start = int(round((t_start - t0) / total_duration * (T_a - 1)))
                    a_end = int(round((t_end - t0) / total_duration * (T_a - 1)))
                    a_start = max(0, min(a_start, T_a - 1))
                    a_end = max(a_start, min(a_end, T_a - 1))
                    audio_seg = audio_emb[a_start:a_end + 1]
            else:
                if orig_T_v > 0:
                    a_start = int(np.floor(start / orig_T_v * T_a))
                    a_end = int(np.ceil(end / orig_T_v * T_a)) - 1
                    a_start = max(0, min(a_start, T_a - 1))
                    a_end = max(a_start, min(a_end, T_a - 1))
                    audio_seg = audio_emb[a_start:a_end + 1]

            if len(audio_seg) != T_v:
                audio_emb = self._interpolate_audio(audio_seg, T_v)
            else:
                audio_emb = audio_seg
        
        # Video-level label (any fake frame -> fake video)
        video_label = 1 if frame_labels.sum() > 0 else 0
        
        return {
            'visual': torch.from_numpy(visual_emb).float(),  # [T, 512]
            'audio': torch.from_numpy(audio_emb).float(),    # [T, 1024]
            'frame_labels': torch.from_numpy(frame_labels).long(),  # [T]
            'video_label': torch.tensor(video_label, dtype=torch.long),
            'video_id': sample['video_id'],
            'num_frames': T_v
        }
    
    def _interpolate_audio(self, audio_emb: np.ndarray, target_len: int) -> np.ndarray:
        """Interpolate audio embeddings to match visual length"""
        T_a, D = audio_emb.shape
        
        if T_a == 0:
            return np.zeros((target_len, D), dtype=audio_emb.dtype)
        if T_a == 1:
            return np.repeat(audio_emb, target_len, axis=0)

        # Linear interpolation
        indices = np.linspace(0, T_a - 1, target_len)
        
        interpolated = np.zeros((target_len, D), dtype=audio_emb.dtype)
        for i, idx in enumerate(indices):
            idx_low = int(np.floor(idx))
            idx_high = min(int(np.ceil(idx)), T_a - 1)
            weight = idx - idx_low
            
            interpolated[i] = (1 - weight) * audio_emb[idx_low] + weight * audio_emb[idx_high]
        
        return interpolated


def collate_variable_length(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for variable-length sequences
    Pads to max length in batch and creates attention mask
    """
    # Find max length in batch
    max_len = max(item['num_frames'] for item in batch)
    batch_size = len(batch)
    
    # Infer feature dims from batch to avoid hardcoding (audio can be 512-d or 1024-d)
    visual_dim = int(batch[0]['visual'].size(-1))
    audio_dim = int(batch[0]['audio'].size(-1))

    # Initialize padded tensors
    visual_padded = torch.zeros(batch_size, max_len, visual_dim)
    audio_padded = torch.zeros(batch_size, max_len, audio_dim)
    frame_labels_padded = torch.zeros(batch_size, max_len, dtype=torch.long)
    mask = torch.ones(batch_size, max_len, dtype=torch.bool)  # True = padding
    
    video_labels = []
    video_ids = []
    
    for i, item in enumerate(batch):
        T = item['num_frames']
        visual_padded[i, :T] = item['visual']
        audio_padded[i, :T] = item['audio']
        frame_labels_padded[i, :T] = item['frame_labels']
        mask[i, :T] = False  # False = valid
        
        video_labels.append(item['video_label'])
        video_ids.append(item['video_id'])
    
    return {
        'visual': visual_padded,
        'audio': audio_padded,
        'frame_labels': frame_labels_padded,
        'video_labels': torch.stack(video_labels),
        'mask': mask,
        'video_ids': video_ids
    }


def get_dataloaders(
    features_root: str,
    splits: List[str] = ['train', 'dev', 'test'],
    batch_size: int = 8,
    num_workers: int = 4,
    max_frames: int = 512,
    event_centric_prob: float = 0.5,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    **kwargs
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for all splits
    
    Args:
        features_root: Path to extracted features
        splits: List of splits to load
        batch_size: Batch size
        num_workers: Number of worker processes
        max_frames: Max frames per video
        event_centric_prob: Event-centric sampling probability for fake videos (train only)
    
    Returns:
        Dictionary of {split: DataLoader}
    """
    dataloaders = {}
    
    for split in splits:
        dataset = LAVDFLocalizationDataset(
            features_root=features_root,
            split=split,
            max_frames=max_frames,
            event_centric_prob=event_centric_prob,
            **kwargs
        )

        sampler = None
        # DDP: only shard TRAIN data. For eval splits, sharding is optional and can
        # lead to partial metrics if only rank0 evaluates.
        if distributed and world_size > 1 and split == 'train':
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=(split == 'train'),
                drop_last=(split == 'train')
            )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False if sampler is not None else (split == 'train'),
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_variable_length,
            pin_memory=True,
            drop_last=(split == 'train'),
            persistent_workers=False,  # Disable to avoid worker hangs on HDD
            timeout=300 if num_workers > 0 else 0,  # 5min timeout for worker
        )
        
        dataloaders[split] = dataloader
        if sampler is not None:
            dataloaders[f"{split}_sampler"] = sampler
    
    # Return pos_weight from train dataset
    if 'train' in splits:
        train_dataset = dataloaders['train'].dataset
        dataloaders['pos_weight'] = train_dataset.pos_weight
    
    return dataloaders


if __name__ == '__main__':
    # Test dataset loading
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_root', type=str, required=True)
    parser.add_argument('--split', type=str, default='dev')
    args = parser.parse_args()
    
    dataset = LAVDFLocalizationDataset(
        features_root=args.features_root,
        split=args.split,
        max_frames=512
    )
    
    print(f"\n[TEST] Loading first sample...")
    sample = dataset[0]
    print(f"Visual: {sample['visual'].shape}")
    print(f"Audio: {sample['audio'].shape}")
    print(f"Frame labels: {sample['frame_labels'].shape}")
    print(f"Video label: {sample['video_label']}")
    print(f"Fake frames: {sample['frame_labels'].sum().item()} / {len(sample['frame_labels'])}")
    
    print(f"\n[TEST] Testing DataLoader...")
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_variable_length)
    batch = next(iter(loader))
    print(f"Batch visual: {batch['visual'].shape}")
    print(f"Batch audio: {batch['audio'].shape}")
    print(f"Batch frame_labels: {batch['frame_labels'].shape}")
    print(f"Batch mask: {batch['mask'].shape}")
    print(f"Batch video_labels: {batch['video_labels']}")
