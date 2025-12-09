import os
import cv2
import sys
from face_extractor import BottomFaceExtractor


def test_single_image(image_path, output_dir="test_output"):
    """
    Test bottom face extraction on a single image.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found - {image_path}")
        return False
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        extractor = BottomFaceExtractor()
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image")
        return False
    
    result = extractor.process_image(image, save_debug=True)
    
    if result['success']:
        aligned_path = os.path.join(output_dir, "aligned_face.png")
        bottom_path = os.path.join(output_dir, "bottom_face.png")
        debug_path = os.path.join(output_dir, "debug.png")
        
        cv2.imwrite(aligned_path, result['aligned_face'])
        cv2.imwrite(bottom_path, result['bottom_face'])
        if result['debug_image'] is not None:
            cv2.imwrite(debug_path, result['debug_image'])
        
        print(f"Success! Results saved to {output_dir}/")
        return True
    else:
        print("Processing failed")
        return False


def test_video(video_path, output_dir="test_video_output", sample_rate=1):
    """
    Test bottom face extraction on video frames.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found - {video_path}")
        return False
    
    try:
        extractor = BottomFaceExtractor()
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    try:
        results = extractor.process_video_frames(
            video_path=video_path,
            output_dir=output_dir,
            sample_rate=sample_rate
        )
        
        success_count = sum(1 for r in results if r['success'])
        total_count = len(results)
        
        import json
        stats_path = os.path.join(output_dir, "statistics.json")
        stats = {
            'total_frames': total_count,
            'success_count': success_count,
            'failed_count': total_count - success_count,
            'sample_rate': sample_rate,
            'results': results
        }
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"Complete! Processed {success_count}/{total_count} frames. Results in {output_dir}/")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """
    Main test function.
    """
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python test_face_extraction.py image <image_path>")
        print("  python test_face_extraction.py video <video_path> [sample_rate]")
        print("\nExamples:")
        print("  python test_face_extraction.py image test.jpg")
        print("  python test_face_extraction.py video test.mp4 10")
        return
    
    mode = sys.argv[1].lower()
    
    if mode == "image" and len(sys.argv) >= 3:
        image_path = sys.argv[2]
        test_single_image(image_path)
        
    elif mode == "video" and len(sys.argv) >= 3:
        video_path = sys.argv[2]
        sample_rate = int(sys.argv[3]) if len(sys.argv) >= 4 else 1
        test_video(video_path, sample_rate=sample_rate)
        
    else:
        print("Error: Invalid arguments")
        print("Usage:")
        print("  python test_face_extraction.py image <image_path>")
        print("  python test_face_extraction.py video <video_path> [sample_rate]")


if __name__ == "__main__":
    main()

