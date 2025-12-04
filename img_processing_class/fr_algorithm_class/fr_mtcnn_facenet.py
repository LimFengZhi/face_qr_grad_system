import os
import cv2 as cv
import numpy as np
import pickle
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.preprocessing import Normalizer

class FaceRecognitionMTCNNFaceNet:
    
    def __init__(self, file_path, threshold=0.5, target_size=(160, 160)):
        self.file_path = file_path
        self.threshold = threshold  # Cosine similarity threshold
        self.target_size = target_size
        self.known_encodings = []
        self.known_ids = []
        
        # Initialize MTCNN detector and FaceNet embedder
        print("Loading MTCNN + FaceNet model...")
        try:
            self.detector = MTCNN()
            self.embedder = FaceNet()
            self.normalizer = Normalizer('l2')
            print(f"✓ MTCNN + FaceNet initialized")
        except Exception as e:
            print(f"Model initialization error: {e}")
        
        self.load_encodings()
    
    def load_encodings(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'rb') as f:
                data = pickle.load(f)
                self.known_encodings = data.get('encodings', [])
                self.known_ids = data.get('ids', [])
                print(f"✓ Loaded {len(self.known_ids)} encodings")
        else:
            self.known_encodings = []
            self.known_ids = []
    
    def save_encoding(self):
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with open(self.file_path, 'wb') as f:
            pickle.dump({
                'encodings': self.known_encodings,
                'ids': self.known_ids
            }, f)
    
    def compute_distance(self, enc1, enc2):
        """Compute cosine distance (lower = more similar)"""
        enc1, enc2 = np.array(enc1), np.array(enc2)
        dot = np.dot(enc1, enc2)
        norm1, norm2 = np.linalg.norm(enc1), np.linalg.norm(enc2)
        if norm1 == 0 or norm2 == 0:
            return 1.0
        similarity = dot / (norm1 * norm2)
        return 1 - similarity  # Convert to distance
    
    def detect_face(self, face_img):
        """
        Detect face in image - similar to HOG+Dlib detect_face
        
        Args:
            face_img: Input image (RGB)
        
        Returns:
            face_region: Dict with box and keypoints, or None if no face
        """
        try:
            faces = self.detector.detect_faces(face_img)
            
            if len(faces) == 0:
                print(" No face detected")
                return None
            
            # Return largest face (by area)
            largest_face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
            return largest_face
            
        except Exception as e:
            print(f" Detection error: {e}")
            return None
    
    def recognise_face(self, face_img, face_region=None):
        """
        
        Args:
            face_img: Input image (RGB)
            face_region: Face region dict from detect_face
        
        Returns:
            encoding: Face embedding array
            success: Boolean indicating success
        """
        try:
            if face_region is None:
                # If no region provided, detect first
                face_region = self.detect_face(face_img)
                if face_region is None:
                    print(" Failed")
                    return None, False
            
            # Extract face coordinates
            x, y, w, h = face_region['box']
            x, y = abs(x), abs(y)
            
            # Ensure bounds are within image
            h_img, w_img = face_img.shape[:2]
            x = max(0, x)
            y = max(0, y)
            x2 = min(w_img, x + w)
            y2 = min(h_img, y + h)
            
            # Extract and resize face
            face = face_img[y:y2, x:x2]
            
            if face.size == 0:
                print(" Failed")
                return None, False
            
            face = cv.resize(face, self.target_size)
            
            # Get embedding
            embedding = self.embedder.embeddings(np.expand_dims(face.astype('float32'), axis=0))[0]
            embedding = self.normalizer.transform([embedding])[0]
            
            print(" Success")
            return embedding, True
            
        except Exception as e:
            print(f" Failed: {e}")
            return None, False
    
    def compare_encoding(self, encoding):
        """
        Compare encoding against known faces
        
        Returns:
            matched_id: ID of best match or None
            distance: Distance to best match
            matched: Boolean indicating if match found
        """
        if len(self.known_encodings) == 0:
            return None, None, False
        
        best_dist = float('inf')
        best_id = None
        
        for known_enc, known_id in zip(self.known_encodings, self.known_ids):
            dist = self.compute_distance(encoding, known_enc)
            if dist < best_dist:
                best_dist = dist
                best_id = known_id
        
        # threshold is similarity, so convert: distance < (1 - threshold)
        if best_dist < (1 - self.threshold):
            return best_id, float(best_dist), True
        else:
            return None, float(best_dist), False
    
    def register_face(self, encoding, person_id):
        """Register encoding with ID"""
        self.known_encodings.append(encoding)
        self.known_ids.append(person_id)
        self.save_encoding()
    
    def clear_encodings(self):
        self.known_encodings = []
        self.known_ids = []
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
        print("✓ Encodings cleared")