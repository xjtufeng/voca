import os
import cv2
import mediapipe as mp
import numpy as np


class BottomFaceExtractor:
    def __init__(self, model_complexity=1, min_detection_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence
        )
    
    def face_center(self, faces_landmarks, x, y, image_width, image_height):
        """
        Select the face closest to target coordinates from multiple detected faces.
        
        Args:
            faces_landmarks: List of MediaPipe face landmarks
            x, y: Target coordinates (normalized 0-1)
            image_width, image_height: Image dimensions
        
        Returns:
            Closest face landmarks
        """
        min_dis = float('inf')
        best_idx = 0
        
        for idx, face_landmarks in enumerate(faces_landmarks):
            nose_landmark = face_landmarks.landmark[1]
            center_x = nose_landmark.x
            center_y = nose_landmark.y
            
            distance = (center_x - x)**2 + (center_y - y)**2
            
            if distance < min_dis:
                min_dis = distance
                best_idx = idx
        
        return faces_landmarks[best_idx]
    
    def detect_face(self, image):
        """
        Detect faces in the image.
        
        Args:
            image: BGR image (OpenCV format)
        
        Returns:
            MediaPipe detection results
        """
        if image is None or image.size == 0:
            return None
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        return results
    
    def align_face(self, image, face_landmarks):
        """
        Align face to standard pose using landmarks.
        
        Args:
            image: BGR image
            face_landmarks: MediaPipe face landmarks
        
        Returns:
            aligned_face: Aligned 256×256 face image
            face_landmarks: Landmarks object
        """
        h, w = image.shape[:2]
        
        left_eye = face_landmarks.landmark[33]
        left_eye_coords = np.array([left_eye.x * w, left_eye.y * h])
        
        right_eye = face_landmarks.landmark[263]
        right_eye_coords = np.array([right_eye.x * w, right_eye.y * h])
        
        nose_tip = face_landmarks.landmark[1]
        nose_coords = np.array([nose_tip.x * w, nose_tip.y * h])
        
        eyes_center = (left_eye_coords + right_eye_coords) / 2.0
        
        dY = right_eye_coords[1] - left_eye_coords[1]
        dX = right_eye_coords[0] - left_eye_coords[0]
        angle = np.degrees(np.arctan2(dY, dX))
        
        desired_eye_dist = 256 * 0.4
        current_eye_dist = np.linalg.norm(right_eye_coords - left_eye_coords)
        scale = desired_eye_dist / current_eye_dist
        
        eyes_center_tuple = (float(eyes_center[0]), float(eyes_center[1]))
        
        M = cv2.getRotationMatrix2D(eyes_center_tuple, float(angle), float(scale))
        
        M[0, 2] += (256 / 2 - eyes_center_tuple[0])
        M[1, 2] += (256 * 0.35 - eyes_center_tuple[1])
        
        aligned_face = cv2.warpAffine(image, M, (256, 256), flags=cv2.INTER_CUBIC)
        
        return aligned_face, face_landmarks
    
    def extract_bottom_face(self, aligned_face, split_point=None):
        """
        Extract bottom face region (below nose bridge center) from aligned face.
        
        Uses MediaPipe landmark 168 (nose bridge center) for precise split line localization.
        This adaptive approach improves accuracy compared to fixed-value methods.
        
        Args:
            aligned_face: Aligned 256×256 face image
            split_point: Split point y coordinate. If None, auto-detect
        
        Returns:
            bottom_face: Bottom face region image
            split_y: Actual split point y coordinate used
        """
        if split_point is None:
            rgb_aligned = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
            rgb_aligned = cv2.cvtColor(rgb_aligned, cv2.COLOR_GRAY2RGB) if len(rgb_aligned.shape) == 2 else cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
            
            results = self.face_mesh.process(rgb_aligned)
            
            if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
                landmarks = results.multi_face_landmarks[0]
                
                nose_bridge = landmarks.landmark[168]
                split_y = int(nose_bridge.y * 256)
                
                split_y = max(100, min(split_y, 150))
                
            else:
                split_y = 120
        else:
            split_y = split_point
        
        bottom_face = aligned_face[split_y:256, 0:256]
        
        return bottom_face, split_y
    
    def process_image(self, image, target_coords=None, save_debug=False):
        """
        Process single image to extract aligned face and bottom face.
        
        Args:
            image: Input image (BGR)
            target_coords: (x, y) Target speaker coordinates (normalized [0,1])
            save_debug: Whether to save debug information
        
        Returns:
            result: {
                'success': bool,
                'aligned_face': np.array (256×256×3),
                'bottom_face': np.array (H×256×3),
                'split_y': int,
                'landmarks': MediaPipe landmarks
            }
        """
        result = {
            'success': False,
            'aligned_face': None,
            'bottom_face': None,
            'split_y': None,
            'landmarks': None,
            'debug_image': None
        }
        
        detection_results = self.detect_face(image)
        
        if detection_results is None or not detection_results.multi_face_landmarks:
            return result
        
        faces_landmarks = detection_results.multi_face_landmarks
        
        if len(faces_landmarks) > 1 and target_coords is not None:
            h, w = image.shape[:2]
            face_landmarks = self.face_center(faces_landmarks, 
                                             target_coords[0], target_coords[1],
                                             w, h)
        else:
            face_landmarks = faces_landmarks[0]
        
        result['landmarks'] = face_landmarks
        
        try:
            aligned_face, _ = self.align_face(image, face_landmarks)
            result['aligned_face'] = aligned_face
        except Exception:
            return result
        
        try:
            bottom_face, split_y = self.extract_bottom_face(aligned_face)
            result['bottom_face'] = bottom_face
            result['split_y'] = split_y
        except Exception:
            return result
        
        if save_debug:
            debug_image = aligned_face.copy()
            cv2.line(debug_image, (0, split_y), (256, split_y), (0, 255, 0), 2)
            cv2.putText(debug_image, f"Split at y={split_y}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            result['debug_image'] = debug_image
        
        result['success'] = True
        return result
    
    def process_video_frames(self, video_path, output_dir, 
                            sample_rate=1, video_list_df=None):
        """
        Process video frames to extract bottom faces.
        
        Args:
            video_path: Video file path
            output_dir: Output directory
            sample_rate: Process every N frames (1 means every frame)
            video_list_df: DataFrame with target coordinates (optional)
        
        Returns:
            results: List of processing results
        """
        face_dir = os.path.join(output_dir, 'aligned_faces')
        bottom_dir = os.path.join(output_dir, 'bottom_faces')
        debug_dir = os.path.join(output_dir, 'debug')
        
        os.makedirs(face_dir, exist_ok=True)
        os.makedirs(bottom_dir, exist_ok=True)
        os.makedirs(debug_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        results = []
        frame_idx = 0
        processed_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                result = self.process_image(frame, save_debug=True)
                
                if result['success']:
                    face_filename = f"frame_{frame_idx:06d}.png"
                    cv2.imwrite(os.path.join(face_dir, face_filename), 
                               result['aligned_face'])
                    cv2.imwrite(os.path.join(bottom_dir, face_filename), 
                               result['bottom_face'])
                    if result['debug_image'] is not None:
                        cv2.imwrite(os.path.join(debug_dir, face_filename),
                                   result['debug_image'])
                    
                    processed_count += 1
                    results.append({
                        'frame_idx': frame_idx,
                        'timestamp': frame_idx / fps,
                        'split_y': result['split_y'],
                        'success': True
                    })
                else:
                    results.append({
                        'frame_idx': frame_idx,
                        'timestamp': frame_idx / fps,
                        'success': False
                    })
            
            frame_idx += 1
        
        cap.release()
        
        return results




