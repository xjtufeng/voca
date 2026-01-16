#!/bin/bash
# Quick start: Check status and start dev extraction on SSD

set -e

cd ~/jhspoolers/voca

LAVDF_ROOT="/hpc2ssd/JH_DATA/spooler/xfeng733/LAV-DF"
OUTPUT_ROOT="/hpc2ssd/JH_DATA/spooler/xfeng733/LAV-DF_feats"

echo "======================================================================="
echo "DEV FEATURE EXTRACTION - QUICK START (SSD)"
echo "======================================================================="
echo ""

# Check if dev split exists
if [ ! -d "$LAVDF_ROOT/dev" ]; then
    echo "❌ ERROR: Dev split not found at $LAVDF_ROOT/dev"
    echo "   Please split the dataset first using scripts/split_dev_from_train.py"
    exit 1
fi

# Count total dev videos
TOTAL_DEV=$(find "$LAVDF_ROOT/dev" -name "*.mp4" 2>/dev/null | wc -l)
echo "✅ Found dev split: $TOTAL_DEV videos"

# Check existing extraction progress
mkdir -p "$OUTPUT_ROOT/dev"
NUM_EXTRACTED=$(find "$OUTPUT_ROOT/dev" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
REMAINING=$((TOTAL_DEV - NUM_EXTRACTED))

echo "   Already extracted: $NUM_EXTRACTED"
echo "   Remaining:         $REMAINING"
echo ""

if [ "$REMAINING" -eq 0 ]; then
    echo "🎉 All dev videos already extracted!"
    exit 0
fi

# Check GPU availability
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "Available GPUs: $NUM_GPUS"
echo ""

# Ask user for extraction method
echo "Choose extraction method:"
echo "  [1] Single GPU (simpler, slower)"
echo "  [2] Dual GPU parallel (2x faster, recommended)"
echo -n "Your choice [1/2]: "
read -r CHOICE

echo ""
echo "======================================================================="

if [ "$CHOICE" == "2" ] && [ "$NUM_GPUS" -ge 2 ]; then
    echo "Starting PARALLEL extraction on 2 GPUs..."
    echo "======================================================================="
    echo ""
    
    # Create temp directory for video lists
    mkdir -p temp_lists
    
    # Get remaining videos
    find "$LAVDF_ROOT/dev" -name "*.mp4" > temp_lists/all_dev_videos.txt
    
    if [ "$NUM_EXTRACTED" -gt 0 ]; then
        # Filter out already extracted
        find "$OUTPUT_ROOT/dev" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort > temp_lists/extracted_ids.txt
        
        while IFS= read -r video_path; do
            video_id=$(basename "$video_path" .mp4)
            if ! grep -q "^${video_id}$" temp_lists/extracted_ids.txt 2>/dev/null; then
                echo "$video_path"
            fi
        done < temp_lists/all_dev_videos.txt > temp_lists/remaining_videos.txt
    else
        cp temp_lists/all_dev_videos.txt temp_lists/remaining_videos.txt
    fi
    
    # Split into 2 parts
    HALF=$(((REMAINING + 1) / 2))
    split -l $HALF temp_lists/remaining_videos.txt temp_lists/dev_part_
    
    mv temp_lists/dev_part_aa temp_lists/dev_gpu0.txt
    [ -f temp_lists/dev_part_ab ] && mv temp_lists/dev_part_ab temp_lists/dev_gpu1.txt || touch temp_lists/dev_gpu1.txt
    
    GPU0_COUNT=$(wc -l < temp_lists/dev_gpu0.txt)
    GPU1_COUNT=$(wc -l < temp_lists/dev_gpu1.txt)
    
    echo "GPU 0 will process: $GPU0_COUNT videos"
    echo "GPU 1 will process: $GPU1_COUNT videos"
    echo ""
    echo "Starting extraction in background..."
    
    # GPU 0
    CUDA_VISIBLE_DEVICES=0 python prepare_features_dataset.py \
        --dataset_root "$LAVDF_ROOT/dev" \
        --output_root "$OUTPUT_ROOT/dev" \
        --video_list temp_lists/dev_gpu0.txt \
        --use_gpu \
        --skip_existing \
        2>&1 | tee logs/dev_extraction_gpu0_$(date +%Y%m%d_%H%M%S).log &
    PID0=$!
    
    # GPU 1
    CUDA_VISIBLE_DEVICES=1 python prepare_features_dataset.py \
        --dataset_root "$LAVDF_ROOT/dev" \
        --output_root "$OUTPUT_ROOT/dev" \
        --video_list temp_lists/dev_gpu1.txt \
        --use_gpu \
        --skip_existing \
        2>&1 | tee logs/dev_extraction_gpu1_$(date +%Y%m%d_%H%M%S).log &
    PID1=$!
    
    echo "✅ Extraction started!"
    echo "   GPU 0 PID: $PID0"
    echo "   GPU 1 PID: $PID1"
    echo ""
    echo "Monitor progress:"
    echo "   tail -f logs/dev_extraction_gpu0_*.log"
    echo "   tail -f logs/dev_extraction_gpu1_*.log"
    echo ""
    echo "Wait for completion:"
    echo "   wait $PID0 && wait $PID1"
    echo ""
    
else
    echo "Starting SINGLE GPU extraction..."
    echo "======================================================================="
    echo ""
    
    python prepare_features_dataset.py \
        --dataset_root "$LAVDF_ROOT/dev" \
        --output_root "$OUTPUT_ROOT/dev" \
        --use_gpu \
        --skip_existing \
        2>&1 | tee logs/dev_extraction_ssd_$(date +%Y%m%d_%H%M%S).log
    
    echo ""
    echo "✅ Extraction complete!"
fi

# Final verification
echo ""
echo "======================================================================="
echo "FINAL STATUS"
echo "======================================================================="
NUM_FINAL=$(find "$OUTPUT_ROOT/dev" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
echo "Total extracted: $NUM_FINAL / $TOTAL_DEV videos"

if [ "$NUM_FINAL" -eq "$TOTAL_DEV" ]; then
    echo "🎉 All dev features extracted successfully!"
else
    STILL_REMAINING=$((TOTAL_DEV - NUM_FINAL))
    echo "⚠️  Still remaining: $STILL_REMAINING videos"
    echo "   You can rerun this script to continue"
fi

