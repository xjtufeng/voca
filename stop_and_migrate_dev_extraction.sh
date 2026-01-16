#!/bin/bash
# Stop dev extraction on 4-GPU machine and continue on 2-GPU SSD machine

set -e

echo "======================================================================="
echo "STOP AND MIGRATE DEV FEATURE EXTRACTION"
echo "======================================================================="
echo ""

# ===================================================================
# PART 1: Stop extraction on 4-GPU HDD machine
# ===================================================================
echo "[PART 1] Stopping dev extraction on 4-GPU HDD machine..."
echo "-----------------------------------------------------------------------"

HDD_DIR="/hpc2hdd/home/xfeng733/LAV-DF_feats"
HDD_DEV_DIR="$HDD_DIR/dev"

if [ -d "$HDD_DEV_DIR" ]; then
    echo "✅ Found dev directory on HDD: $HDD_DEV_DIR"
    
    # Count extracted features
    NUM_EXTRACTED=$(find "$HDD_DEV_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    echo "   Extracted videos: $NUM_EXTRACTED"
    
    # Check for running processes
    echo ""
    echo "Checking for running extraction processes..."
    ps aux | grep -E "prepare_features|extract.*dev" | grep -v grep || echo "   No active extraction processes found"
    
    echo ""
    echo "⚠️  Please manually kill any extraction processes on the 4-GPU machine if needed:"
    echo "   ssh to 4-GPU machine and run: pkill -f 'prepare_features.*dev'"
else
    echo "❌ No dev directory found on HDD: $HDD_DEV_DIR"
fi

echo ""
echo "-----------------------------------------------------------------------"

# ===================================================================
# PART 2: Check SSD environment and prepare for continuation
# ===================================================================
echo "[PART 2] Preparing SSD environment for dev extraction..."
echo "-----------------------------------------------------------------------"

SSD_DIR="/hpc2ssd/JH_DATA/spooler/xfeng733/LAV-DF_feats"
SSD_DEV_DIR="$SSD_DIR/dev"
LAVDF_ROOT="/hpc2ssd/JH_DATA/spooler/xfeng733/LAV-DF"

# Check if dev split info exists
if [ ! -d "$LAVDF_ROOT/dev" ]; then
    echo "❌ ERROR: $LAVDF_ROOT/dev directory not found!"
    echo "   You need to split the dataset first."
    exit 1
fi

echo "✅ Found LAV-DF dev split: $LAVDF_ROOT/dev"

# Count total dev videos
TOTAL_DEV=$(find "$LAVDF_ROOT/dev" -name "*.mp4" 2>/dev/null | wc -l)
echo "   Total dev videos: $TOTAL_DEV"

# Check existing extracted features on SSD
if [ -d "$SSD_DEV_DIR" ]; then
    NUM_SSD_EXTRACTED=$(find "$SSD_DEV_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    echo "   Already extracted on SSD: $NUM_SSD_EXTRACTED"
    REMAINING=$((TOTAL_DEV - NUM_SSD_EXTRACTED))
else
    echo "   No features on SSD yet"
    NUM_SSD_EXTRACTED=0
    REMAINING=$TOTAL_DEV
    mkdir -p "$SSD_DEV_DIR"
fi

echo ""
echo "📊 Summary:"
echo "   Total dev videos:      $TOTAL_DEV"
echo "   Already on SSD:        $NUM_SSD_EXTRACTED"
echo "   Remaining to extract: $REMAINING"

echo ""
echo "-----------------------------------------------------------------------"

# ===================================================================
# PART 3: Generate extraction command for SSD
# ===================================================================
echo "[PART 3] Generating extraction command for SSD environment..."
echo "-----------------------------------------------------------------------"

EXTRACT_SCRIPT="~/jhspoolers/voca/prepare_features_dataset.py"
OUTPUT_CMD="continue_dev_extraction_ssd.sh"

cat > "$OUTPUT_CMD" << 'EXTRACT_EOF'
#!/bin/bash
# Continue dev feature extraction on 2-GPU SSD machine

set -e

cd ~/jhspoolers/voca

LAVDF_ROOT="/hpc2ssd/JH_DATA/spooler/xfeng733/LAV-DF"
OUTPUT_ROOT="/hpc2ssd/JH_DATA/spooler/xfeng733/LAV-DF_feats"

echo "======================================================================="
echo "CONTINUE DEV FEATURE EXTRACTION (SSD)"
echo "======================================================================="
echo ""

# Check existing progress
if [ -d "$OUTPUT_ROOT/dev" ]; then
    NUM_DONE=$(find "$OUTPUT_ROOT/dev" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    echo "Already extracted: $NUM_DONE videos"
    echo ""
fi

# Extract dev split (will automatically skip existing)
echo "[INFO] Starting dev feature extraction..."
echo "  Video root: $LAVDF_ROOT/dev"
echo "  Output: $OUTPUT_ROOT/dev"
echo ""

python prepare_features_dataset.py \
    --dataset_root "$LAVDF_ROOT/dev" \
    --output_root "$OUTPUT_ROOT/dev" \
    --use_gpu \
    --skip_existing \
    2>&1 | tee logs/dev_extraction_ssd_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "✅ Dev extraction complete!"

# Verify
NUM_EXTRACTED=$(find "$OUTPUT_ROOT/dev" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
echo "Total extracted: $NUM_EXTRACTED videos"
EXTRACT_EOF

chmod +x "$OUTPUT_CMD"

echo "✅ Generated: $OUTPUT_CMD"
echo ""
echo "To run the extraction, execute:"
echo "  bash $OUTPUT_CMD"
echo ""

# ===================================================================
# PART 4: Generate parallel extraction command (for faster processing)
# ===================================================================
echo "[PART 4] Generating parallel extraction command (optional, faster)..."
echo "-----------------------------------------------------------------------"

PARALLEL_CMD="parallel_dev_extraction_ssd.sh"

cat > "$PARALLEL_CMD" << 'PARALLEL_EOF'
#!/bin/bash
# Parallel dev feature extraction on 2-GPU SSD machine

set -e

cd ~/jhspoolers/voca

LAVDF_ROOT="/hpc2ssd/JH_DATA/spooler/xfeng733/LAV-DF"
OUTPUT_ROOT="/hpc2ssd/JH_DATA/spooler/xfeng733/LAV-DF_feats"

echo "======================================================================="
echo "PARALLEL DEV FEATURE EXTRACTION (2 GPUs)"
echo "======================================================================="
echo ""

# Create temp directory for video lists
mkdir -p temp_lists

# Get all dev videos
find "$LAVDF_ROOT/dev" -name "*.mp4" > temp_lists/all_dev_videos.txt
TOTAL=$(wc -l < temp_lists/all_dev_videos.txt)

# Filter out already extracted
if [ -d "$OUTPUT_ROOT/dev" ]; then
    # Get list of extracted video IDs
    find "$OUTPUT_ROOT/dev" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort > temp_lists/extracted_ids.txt
    
    # Filter: only keep videos not yet extracted
    while IFS= read -r video_path; do
        video_id=$(basename "$video_path" .mp4)
        if ! grep -q "^${video_id}$" temp_lists/extracted_ids.txt; then
            echo "$video_path"
        fi
    done < temp_lists/all_dev_videos.txt > temp_lists/remaining_videos.txt
    
    REMAINING=$(wc -l < temp_lists/remaining_videos.txt)
    echo "Already extracted: $((TOTAL - REMAINING)) videos"
    echo "Remaining: $REMAINING videos"
else
    cp temp_lists/all_dev_videos.txt temp_lists/remaining_videos.txt
    REMAINING=$TOTAL
    echo "Total to extract: $REMAINING videos"
fi

if [ "$REMAINING" -eq 0 ]; then
    echo "✅ All dev videos already extracted!"
    exit 0
fi

# Split into 2 parts for 2 GPUs
HALF=$((REMAINING / 2))
split -l $HALF temp_lists/remaining_videos.txt temp_lists/dev_part_

mv temp_lists/dev_part_aa temp_lists/dev_gpu0.txt
mv temp_lists/dev_part_ab temp_lists/dev_gpu1.txt

echo ""
echo "GPU 0: $(wc -l < temp_lists/dev_gpu0.txt) videos"
echo "GPU 1: $(wc -l < temp_lists/dev_gpu1.txt) videos"
echo ""

# Extract function
extract_batch() {
    local gpu_id=$1
    local video_list=$2
    
    echo "[GPU $gpu_id] Starting extraction..."
    
    CUDA_VISIBLE_DEVICES=$gpu_id python prepare_features_dataset.py \
        --dataset_root "$LAVDF_ROOT/dev" \
        --output_root "$OUTPUT_ROOT/dev" \
        --video_list "$video_list" \
        --use_gpu \
        --skip_existing \
        2>&1 | tee logs/dev_extraction_gpu${gpu_id}_$(date +%Y%m%d_%H%M%S).log
    
    echo "[GPU $gpu_id] ✅ Done!"
}

# Run in parallel
extract_batch 0 temp_lists/dev_gpu0.txt &
PID0=$!

extract_batch 1 temp_lists/dev_gpu1.txt &
PID1=$!

# Wait for both to complete
echo "Waiting for both GPUs to complete..."
wait $PID0
wait $PID1

echo ""
echo "✅ Parallel extraction complete!"

# Cleanup
rm -rf temp_lists

# Verify
NUM_EXTRACTED=$(find "$OUTPUT_ROOT/dev" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
echo "Total dev features: $NUM_EXTRACTED"
PARALLEL_EOF

chmod +x "$PARALLEL_CMD"

echo "✅ Generated: $PARALLEL_CMD (2x faster with 2 GPUs)"
echo ""

# ===================================================================
# Summary
# ===================================================================
echo "======================================================================="
echo "SETUP COMPLETE"
echo "======================================================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Stop extraction on 4-GPU machine (if running):"
echo "   ssh to 4-GPU and run: pkill -f 'prepare_features.*dev'"
echo ""
echo "2. Choose extraction method on SSD (2-GPU machine):"
echo ""
echo "   Option A - Single process (simpler):"
echo "     bash $OUTPUT_CMD"
echo ""
echo "   Option B - Parallel 2 GPUs (2x faster, recommended):"
echo "     bash $PARALLEL_CMD"
echo ""
echo "3. Monitor progress:"
echo "   tail -f logs/dev_extraction_*.log"
echo ""

