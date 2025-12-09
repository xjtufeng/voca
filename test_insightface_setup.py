"""
Quick test script to verify InsightFace installation
Run this in voca-insight conda environment
"""
import sys

print("="*60)
print("Testing InsightFace Installation")
print("="*60)

# 1. Test imports
print("\n1. Testing imports...")
try:
    import torch
    print(f"  torch: {torch.__version__}")
except ImportError as e:
    print(f"  ERROR: torch not installed: {e}")
    sys.exit(1)

try:
    from insightface.app import FaceAnalysis
    print(f"  insightface.app: OK")
except ImportError as e:
    print(f"  ERROR: insightface not installed: {e}")
    sys.exit(1)

try:
    import cv2
    print(f"  opencv: {cv2.__version__}")
except ImportError as e:
    print(f"  ERROR: opencv not installed: {e}")
    sys.exit(1)

try:
    import onnxruntime as ort
    print(f"  onnxruntime: {ort.__version__}")
except ImportError as e:
    print(f"  ERROR: onnxruntime not installed: {e}")
    sys.exit(1)

print("  All imports successful!")

# 2. Test InsightFace model loading
print("\n2. Testing InsightFace model loading...")
try:
    from insightface.app import FaceAnalysis
    
    print("  Initializing FaceAnalysis (buffalo_l)...")
    print("  (First time will download ~200MB model, please wait...)")
    
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=-1, det_size=(256, 256))  # CPU mode
    
    print("  FaceAnalysis loaded successfully!")
    
except Exception as e:
    print(f"  ERROR loading FaceAnalysis: {e}")
    sys.exit(1)

# 3. Test on a sample image
print("\n3. Testing on sample image...")
import os

test_img_path = "test_video_output/bottom_faces/frame_000000.png"
if not os.path.exists(test_img_path):
    print(f"  WARNING: Test image not found: {test_img_path}")
    print("  Skipping image test.")
else:
    try:
        img = cv2.imread(test_img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        faces = app.get(img_rgb)
        print(f"  Detected {len(faces)} face(s)")
        
        if len(faces) > 0:
            emb = faces[0].embedding
            print(f"  Embedding shape: {emb.shape}")
            print(f"  Embedding norm: {(emb**2).sum()**0.5:.4f}")
            print("  Sample image test successful!")
        else:
            print("  WARNING: No face detected (might be expected for bottom-face)")
            
    except Exception as e:
        print(f"  ERROR processing image: {e}")

print("\n" + "="*60)
print("InsightFace setup verification complete!")
print("You can now run: python face_encoder_insightface.py")
print("="*60)

