"""
Test video path mapping logic for three-branch dataset
"""
from pathlib import Path

def test_video_path_mapping():
    """Test different video path mapping scenarios"""
    
    # Simulate feature paths (as they appear on HPC)
    test_cases = [
        {
            'name': 'Simple structure (no shard)',
            'features_root': Path('/hpc2ssd/FakeAV_feats'),
            'video_root': Path('/hpc2ssd/FakeAV_stage'),
            'feature_file': Path('/hpc2ssd/FakeAV_feats/fake/00347_id07200_quCElxMhW6g/visual_embeddings.npz'),
            'expected_candidates': [
                Path('/hpc2ssd/FakeAV_stage/fake/00347_id07200_quCElxMhW6g/visual_embeddings.mp4'),
                Path('/hpc2ssd/FakeAV_stage/fake/00347_id07200_quCElxMhW6g.mp4'),
                Path('/hpc2ssd/FakeAV_stage/00347_id07200_quCElxMhW6g.mp4'),
            ]
        },
        {
            'name': 'With shard directory',
            'features_root': Path('/hpc2ssd/FakeAV_feats'),
            'video_root': Path('/hpc2ssd/FakeAV_stage'),
            'feature_file': Path('/hpc2ssd/FakeAV_feats/fake/FakeAV_feats_ab/fake/00025_id01004_867Wlj7Gw68/similarity_stats.npz'),
            'expected_candidates': [
                Path('/hpc2ssd/FakeAV_stage/fake/FakeAV_feats_ab/fake/00025_id01004_867Wlj7Gw68/similarity_stats.mp4'),
                Path('/hpc2ssd/FakeAV_stage/fake/FakeAV_feats_ab/fake/00025_id01004_867Wlj7Gw68.mp4'),
                Path('/hpc2ssd/FakeAV_stage/fake/00025_id01004_867Wlj7Gw68/similarity_stats.mp4'),
                Path('/hpc2ssd/FakeAV_stage/fake/00025_id01004_867Wlj7Gw68.mp4'),
                Path('/hpc2ssd/FakeAV_stage/00025_id01004_867Wlj7Gw68.mp4'),
            ]
        },
    ]
    
    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f"Test: {test_case['name']}")
        print(f"{'='*60}")
        print(f"Feature file: {test_case['feature_file']}")
        print(f"Features root: {test_case['features_root']}")
        print(f"Video root: {test_case['video_root']}")
        
        # Simulate the mapping logic from dataset_three_branch.py (CURRENT VERSION)
        feat_file = test_case['feature_file']
        features_root = test_case['features_root']
        video_root = test_case['video_root']
        
        rel_path = feat_file.relative_to(features_root)
        print(f"\nRelative path: {rel_path}")
        print(f"Relative parts: {rel_path.parts}")
        
        rel_parts = rel_path.parts
        candidates = []
        
        # Common case: video_root/label/<vid>.mp4
        video_dir = rel_path.parent.name
        label_name = rel_parts[0] if rel_parts else ''
        candidates.append(video_root / label_name / f"{video_dir}.mp4")
        # Case: video_root/<vid>.mp4 (no label subdir)
        candidates.append(video_root / f"{video_dir}.mp4")
        # Variant: video_root/label/<vid>/<vid>.mp4
        candidates.append(video_root / label_name / video_dir / f"{video_dir}.mp4")
        # Original recursive path (may include shard directories)
        candidates.append(video_root / rel_path.parent / (feat_file.stem + '.mp4'))
        # If path contains shard like FakeAV_feats_*, try removing that shard
        parent_parts = list(rel_path.parent.parts)
        if len(parent_parts) >= 3 and parent_parts[1].startswith("FakeAV_feats_"):
            parent_no_shard = Path(*([parent_parts[0]] + parent_parts[2:]))
            candidates.append(video_root / parent_no_shard / (feat_file.stem + '.mp4'))
            candidates.append(video_root / parent_no_shard / f"{video_dir}.mp4")
        
        print(f"\nGenerated candidates:")
        for i, cand in enumerate(candidates, 1):
            print(f"  {i}. {cand}")
        
        print(f"\nExpected candidates:")
        for i, exp in enumerate(test_case['expected_candidates'], 1):
            print(f"  {i}. {exp}")
        
        # Check if all expected candidates are in generated
        missing = []
        for exp in test_case['expected_candidates']:
            if exp not in candidates:
                missing.append(exp)
        
        if missing:
            print(f"\n[MISSING] candidates:")
            for m in missing:
                print(f"  - {m}")
        else:
            print(f"\n[OK] All expected candidates generated")

if __name__ == '__main__':
    test_video_path_mapping()

