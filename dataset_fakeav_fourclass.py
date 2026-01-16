#!/usr/bin/env python3
"""
FakeAVCeleb Four-Class Dataset for MRDF Protocol
Supports identity-independent 5-fold cross-validation with 1:1:1:1 balanced sampling
"""
import json
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Tuple
import cv2


class FakeAVCelebFourClassDataset(Dataset):
    """
    FakeAVCeleb Four-Class Dataset
    
    Labels:
        0: FAFV (Fake Audio + Fake Video)
        1: FARV (Fake Audio + Real Video)
        2: RAFV (Real Audio + Fake Video)
        3: RARV (Real Audio + Real Video)
    """
    def __init__(
        self,
        features_root: str,
        split_file: str,  # Path to fold_X_train.json or fold_X_test.json
        max_frames: int = 256,
        load_video_frames: bool = False,
        video_root: Optional[str] = None,
        frame_size: int = 224,
    ):
        """
        Args:
            features_root: Path to FakeAV_feats directory
            split_file: Path to JSON file containing video list for this split
            max_frames: Maximum frames per video
            load_video_frames: Whether to load raw video frames (for CLIP)
            video_root: Path to original videos (if load_video_frames=True)
            frame_size: Target frame size for video frames
        """
        self.features_root = Path(features_root)
        self.split_file = Path(split_file)
        self.max_frames = max_frames
        self.load_video_frames = load_video_frames
        self.video_root = Path(video_root) if video_root else None
        self.frame_size = frame_size
        
        # Load samples from split file
        self.samples = self._load_samples()
        
        # Compute class weights for balanced sampling
        self._compute_class_info()
        
        print(f"[INFO] Loaded {len(self.samples)} samples from {self.split_file.name}")
        print(f"[INFO] Class distribution: FAFV={self.class_counts[0]}, "
              f"FARV={self.class_counts[1]}, RAFV={self.class_counts[2]}, RARV={self.class_counts[3]}")
    
    def _load_samples(self) -> List[Dict]:
        """Load samples from split JSON file"""
        if not self.split_file.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_file}")
        
        with open(self.split_file, 'r') as f:
            samples = json.load(f)
        
        # Verify features exist and add full paths
        valid_samples = []
        for sample in samples:
            # Construct full paths
            video_path_rel = Path(sample['path'])
            audio_path = self.features_root / video_path_rel / "audio_embeddings.npz"
            visual_path = self.features_root / video_path_rel / "visual_embeddings.npz"
            
            if not audio_path.exists() or not visual_path.exists():
                continue
            
            sample['audio_path'] = audio_path
            sample['visual_path'] = visual_path
            valid_samples.append(sample)
        
        if len(valid_samples) < len(samples):
            print(f"[WARN] {len(samples) - len(valid_samples)} samples skipped (missing features)")
        
        return valid_samples
    
    def _compute_class_info(self):
        """Compute class counts and sampling weights"""
        self.class_counts = [0, 0, 0, 0]
        for sample in self.samples:
            self.class_counts[sample['label']] += 1
        
        # Compute per-sample weight for balanced sampling (1:1:1:1)
        self.sample_weights = []
        for sample in self.samples:
            label = sample['label']
            # Weight = 1 / class_count (inversely proportional to class size)
            weight = 1.0 / self.class_counts[label] if self.class_counts[label] > 0 else 0.0
            self.sample_weights.append(weight)
    
    def get_balanced_sampler(self) -> WeightedRandomSampler:
        """
        Create a WeightedRandomSampler for 1:1:1:1 balanced sampling
        
        Returns:
            WeightedRandomSampler that samples each class with equal probability
        """
        return WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=len(self.samples),
            replacement=True  # Allow replacement for balancing
        )
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load audio features
        audio_data = np.load(sample['audio_path'])
        audio_embeds = audio_data['embeddings']  # [T, D]
        
        # Load visual features
        visual_data = np.load(sample['visual_path'])
        visual_embeds = visual_data['embeddings']  # [T, D]
        
        # Ensure same temporal length (should already be aligned)
        T = min(len(audio_embeds), len(visual_embeds), self.max_frames)
        audio_embeds = audio_embeds[:T]
        visual_embeds = visual_embeds[:T]
        
        # Pad if needed
        if T < self.max_frames:
            audio_pad = np.zeros((self.max_frames - T, audio_embeds.shape[1]), dtype=audio_embeds.dtype)
            visual_pad = np.zeros((self.max_frames - T, visual_embeds.shape[1]), dtype=visual_embeds.dtype)
            
            audio_embeds = np.concatenate([audio_embeds, audio_pad], axis=0)
            visual_embeds = np.concatenate([visual_embeds, visual_pad], axis=0)
            
            # Mask for valid frames
            mask = np.ones(self.max_frames, dtype=bool)
            mask[:T] = False  # False = valid, True = padding
        else:
            mask = np.zeros(self.max_frames, dtype=bool)
        
        # Convert to tensors
        audio_tensor = torch.from_numpy(audio_embeds).float()
        visual_tensor = torch.from_numpy(visual_embeds).float()
        mask_tensor = torch.from_numpy(mask)
        label_tensor = torch.tensor(sample['label'], dtype=torch.long)
        
        return {
            'audio': audio_tensor,
            'visual': visual_tensor,
            'mask': mask_tensor,
            'label': label_tensor,
            'video_id': sample['video_id'],
            'identity': sample['identity']
        }


def get_fakeav_dataloaders(
    features_root: str,
    fold_id: int,
    splits_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    max_frames: int = 256,
    balanced_train: bool = True,
) -> Dict[str, DataLoader]:
    """
    Create train/test dataloaders for a specific fold
    
    Args:
        features_root: Path to FakeAV_feats
        fold_id: Fold ID (0-4)
        splits_dir: Directory containing fold split JSON files
        batch_size: Batch size
        num_workers: Number of data loading workers
        max_frames: Maximum frames per video
        balanced_train: Use balanced sampling for training
    
    Returns:
        Dictionary with 'train' and 'test' DataLoaders
    """
    splits_dir = Path(splits_dir)
    
    # Load train dataset
    train_split_file = splits_dir / f'fold_{fold_id}_train.json'
    train_dataset = FakeAVCelebFourClassDataset(
        features_root=features_root,
        split_file=train_split_file,
        max_frames=max_frames
    )
    
    # Load test dataset
    test_split_file = splits_dir / f'fold_{fold_id}_test.json'
    test_dataset = FakeAVCelebFourClassDataset(
        features_root=features_root,
        split_file=test_split_file,
        max_frames=max_frames
    )
    
    # Create dataloaders
    if balanced_train:
        train_sampler = train_dataset.get_balanced_sampler()
        train_shuffle = False  # Sampler handles shuffling
    else:
        train_sampler = None
        train_shuffle = True
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'test': test_loader
    }


if __name__ == '__main__':
    # Test dataset loading
    print("Testing FakeAVCeleb Four-Class Dataset...")
    
    features_root = "/hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_feats"
    splits_dir = "data/fakeav_5fold_splits"
    
    # Test fold 0
    dataloaders = get_fakeav_dataloaders(
        features_root=features_root,
        fold_id=0,
        splits_dir=splits_dir,
        batch_size=8,
        num_workers=0,
        max_frames=256,
        balanced_train=True
    )
    
    print("\n=== Testing Train Loader ===")
    train_batch = next(iter(dataloaders['train']))
    print(f"Audio shape: {train_batch['audio'].shape}")
    print(f"Visual shape: {train_batch['visual'].shape}")
    print(f"Labels: {train_batch['label']}")
    print(f"Label distribution in batch: {torch.bincount(train_batch['label'])}")
    
    print("\n=== Testing Test Loader ===")
    test_batch = next(iter(dataloaders['test']))
    print(f"Audio shape: {test_batch['audio'].shape}")
    print(f"Visual shape: {test_batch['visual'].shape}")
    print(f"Labels: {test_batch['label']}")
    
    print("\n✅ Dataset loading test passed!")

