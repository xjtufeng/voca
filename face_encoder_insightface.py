"""
Face encoder using InsightFace (ArcFace)
- Uses InsightFace's FaceAnalysis (buffalo_l)
- Takes aligned full face or bottom-face crops
- Outputs 512-D L2-normalized embeddings
"""

import os
import glob
import cv2
import numpy as np

from insightface.app import FaceAnalysis


class InsightFaceBottomEncoder:
    """
    Bottom-face encoder based on InsightFace ArcFace
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        ctx_id: int = 0,
        det_size=(256, 256),
    ):
        """
        Initialize InsightFace FaceAnalysis

        Args:
            model_name: InsightFace model pack name (e.g., 'buffalo_l')
            ctx_id: GPU id, use -1 for CPU, 0 for first GPU
            det_size: detection input size
        """
        print("=" * 60)
        print("Initializing InsightFace FaceAnalysis")
        print("=" * 60)

        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

        print(f"  Model: {model_name}")
        print(f"  ctx_id: {ctx_id}")
        print(f"  det_size: {det_size}")
        print("InsightFace FaceAnalysis initialized.\n")

    @staticmethod
    def pad_to_square(img: np.ndarray) -> np.ndarray:
        """
        Pad bottom-face image to square, without stretching.

        Args:
            img: (H, W, 3) BGR image, typically ~156x256 for bottom-face

        Returns:
            padded: square image (max(H, W), max(H, W), 3)
        """
        h, w = img.shape[:2]
        if h == w:
            return img

        if h < w:
            pad_total = w - h
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
            padded = cv2.copyMakeBorder(
                img, pad_top, pad_bottom, 0, 0,
                borderType=cv2.BORDER_CONSTANT,
                value=[0, 0, 0],
            )
        else:
            pad_total = h - w
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            padded = cv2.copyMakeBorder(
                img, 0, 0, pad_left, pad_right,
                borderType=cv2.BORDER_CONSTANT,
                value=[0, 0, 0],
            )

        return padded

    def encode_image(self, img, is_bottom_face: bool = True):
        """
        Encode a single image into 512-D embedding.

        Args:
            img: BGR image (H, W, 3) or path
            is_bottom_face: if True, pad to square before feeding

        Returns:
            embedding: (512,) L2-normalized, or None if no face detected
        """
        if isinstance(img, str):
            img = cv2.imread(img)
            if img is None:
                raise ValueError(f"Cannot read image: {img}")

        if is_bottom_face:
            img = self.pad_to_square(img)

        # InsightFace expects RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faces = self.app.get(img_rgb)
        if len(faces) == 0:
            return None

        # Choose the largest face (should be the only one)
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = face.embedding.astype(np.float32)

        # L2 normalize
        norm = np.linalg.norm(emb) + 1e-8
        emb = emb / norm

        return emb

    def encode_directory(
        self,
        img_dir: str,
        pattern: str = "frame_*.png",
        is_bottom_face: bool = True,
    ):
        """
        Encode all images in a directory.

        Args:
            img_dir: directory containing images
            pattern: glob pattern
            is_bottom_face: whether images are bottom-face crops

        Returns:
            embeddings: (N, 512) array
            paths: list of image paths actually encoded
        """
        paths = sorted(glob.glob(os.path.join(img_dir, pattern)))
        if len(paths) == 0:
            raise ValueError(f"No images found in {img_dir} with pattern {pattern}")

        print(f"Found {len(paths)} images in {img_dir}")
        print("Extracting InsightFace embeddings...")

        embs = []
        valid_paths = []

        for i, p in enumerate(paths):
            try:
                emb = self.encode_image(p, is_bottom_face=is_bottom_face)
                if emb is None:
                    print(f"  WARNING: No face detected in {p}, skipped.")
                    continue
                embs.append(emb)
                valid_paths.append(p)

                if (i + 1) % 50 == 0 or (i + 1) == len(paths):
                    print(f"  Processed {i+1}/{len(paths)} images")

            except Exception as e:
                print(f"  ERROR processing {p}: {e}")

        if len(embs) == 0:
            raise RuntimeError("No valid embeddings extracted.")

        embs = np.stack(embs, axis=0)
        print(f"\nExtracted {len(embs)} embeddings.")
        print(f"  Shape: {embs.shape}")
        print(f"  Mean: {embs.mean():.6f}")
        print(f"  Std:  {embs.std():.6f}")

        return embs, valid_paths


if __name__ == "__main__":
    print("=" * 60)
    print("Testing InsightFace Bottom-Face Encoder on test2")
    print("=" * 60)

    # 1. Initialize encoder (ctx_id=0 for GPU, -1 for CPU)
    encoder = InsightFaceBottomEncoder(
        model_name="buffalo_l",
        ctx_id=0,          # Change to -1 if no GPU
        det_size=(256, 256),
    )

    # 2. Bottom-face directory
    bottom_dir = "test_video_output/bottom_faces"
    if not os.path.isdir(bottom_dir):
        print(f"\nERROR: Bottom-face directory not found: {bottom_dir}")
        print("Please run face extraction first, e.g.:")
        print("  python test_face_extraction.py video test2.mp4 1 test_video_output")
        raise SystemExit(1)

    # 3. Encode all frames
    embeddings, paths = encoder.encode_directory(
        bottom_dir,
        pattern="frame_*.png",
        is_bottom_face=True,
    )

    # 4. Save to npz
    out_path = "test2_visual_embeddings_insightface.npz"
    np.savez(out_path, embeddings=embeddings, paths=paths)
    print(f"\nSaved embeddings to: {out_path}")

    # 5. Simple similarity check
    if embeddings.shape[0] >= 2:
        from sklearn.metrics.pairwise import cosine_similarity

        sim_01 = cosine_similarity(embeddings[0:1], embeddings[1:2])[0, 0]
        sim_0_last = cosine_similarity(embeddings[0:1], embeddings[-1:])[0, 0]

        print("\nValidation:")
        print(f"  Cosine(frame 0 vs 1):   {sim_01:.4f}")
        print(f"  Cosine(frame 0 vs last): {sim_0_last:.4f}")

    print("\n" + "=" * 60)
    print("InsightFace encoder test finished.")
    print("=" * 60)
