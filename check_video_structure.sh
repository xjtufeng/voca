#!/bin/bash
# Check actual video structure on HPC

echo "=== Checking video directory structure ==="
echo ""
echo "1. Check if videos are in /fake/<video_id>.mp4 format:"
find /hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_stage/fake -maxdepth 1 -name "*.mp4" 2>/dev/null | head -3

echo ""
echo "2. Check if videos are in /fake/<video_id>/<video_id>.mp4 format:"
find /hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_stage/fake -maxdepth 2 -mindepth 2 -name "*.mp4" 2>/dev/null | head -3

echo ""
echo "3. Check total video count:"
find /hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_stage -name "*.mp4" 2>/dev/null | wc -l

echo ""
echo "4. Sample video paths (first 5):"
find /hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_stage -name "*.mp4" 2>/dev/null | head -5

echo ""
echo "5. Check for specific video from features:"
echo "Looking for video corresponding to: 00347_id07200_quCElxMhW6g"
find /hpc2ssd/JH_DATA/spooler/xfeng733/FakeAV_stage -name "*00347_id07200_quCElxMhW6g*" 2>/dev/null

echo ""
echo "Done!"

