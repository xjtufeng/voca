#!/bin/bash
# Run dev extraction in background (dual GPU)

cd ~/jhspoolers/voca

LAVDF_ROOT="/hpc2ssd/JH_DATA/spooler/xfeng733/LAV-DF"
OUTPUT_ROOT="/hpc2ssd/JH_DATA/spooler/xfeng733/LAV-DF_feats"

echo "======================================================================="
echo "Starting DEV extraction in BACKGROUND (2 GPUs)"
echo "======================================================================="

# Create log directory
mkdir -p logs

# Check remaining videos
TOTAL_DEV=$(find "$LAVDF_ROOT/dev" -name "*.mp4" 2>/dev/null | wc -l)
NUM_EXTRACTED=$(find "$OUTPUT_ROOT/dev" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
REMAINING=$((TOTAL_DEV - NUM_EXTRACTED))

echo "Total dev videos:  $TOTAL_DEV"
echo "Already extracted: $NUM_EXTRACTED"
echo "Remaining:         $REMAINING"
echo ""

if [ "$REMAINING" -eq 0 ]; then
    echo "✅ All dev videos already extracted!"
    exit 0
fi

# Prepare video lists
mkdir -p temp_lists
find "$LAVDF_ROOT/dev" -name "*.mp4" > temp_lists/all_dev_videos.txt

if [ "$NUM_EXTRACTED" -gt 0 ]; then
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

# Split for 2 GPUs
HALF=$(((REMAINING + 1) / 2))
split -l $HALF temp_lists/remaining_videos.txt temp_lists/dev_part_

mv temp_lists/dev_part_aa temp_lists/dev_gpu0.txt
[ -f temp_lists/dev_part_ab ] && mv temp_lists/dev_part_ab temp_lists/dev_gpu1.txt || touch temp_lists/dev_gpu1.txt

GPU0_COUNT=$(wc -l < temp_lists/dev_gpu0.txt)
GPU1_COUNT=$(wc -l < temp_lists/dev_gpu1.txt)

echo "GPU 0: $GPU0_COUNT videos"
echo "GPU 1: $GPU1_COUNT videos"
echo ""

# Start extraction in background
LOG0="logs/dev_extraction_gpu0_$(date +%Y%m%d_%H%M%S).log"
LOG1="logs/dev_extraction_gpu1_$(date +%Y%m%d_%H%M%S).log"

nohup bash -c "
    CUDA_VISIBLE_DEVICES=0 python prepare_features_dataset.py \
        --dataset_root '$LAVDF_ROOT/dev' \
        --output_root '$OUTPUT_ROOT/dev' \
        --video_list temp_lists/dev_gpu0.txt \
        --use_gpu \
        --skip_existing
" > "$LOG0" 2>&1 &
PID0=$!

nohup bash -c "
    CUDA_VISIBLE_DEVICES=1 python prepare_features_dataset.py \
        --dataset_root '$LAVDF_ROOT/dev' \
        --output_root '$OUTPUT_ROOT/dev' \
        --video_list temp_lists/dev_gpu1.txt \
        --use_gpu \
        --skip_existing
" > "$LOG1" 2>&1 &
PID1=$!

echo "✅ Extraction started in background!"
echo ""
echo "Process IDs:"
echo "  GPU 0: $PID0"
echo "  GPU 1: $PID1"
echo ""
echo "Log files:"
echo "  GPU 0: $LOG0"
echo "  GPU 1: $LOG1"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOG0"
echo "  tail -f $LOG1"
echo ""
echo "Check if still running:"
echo "  ps aux | grep -E '$PID0|$PID1'"
echo ""
echo "Kill if needed:"
echo "  kill $PID0 $PID1"
echo ""
echo "Check extraction progress:"
echo "  watch -n 10 'find $OUTPUT_ROOT/dev -mindepth 1 -maxdepth 1 -type d | wc -l'"
echo ""

