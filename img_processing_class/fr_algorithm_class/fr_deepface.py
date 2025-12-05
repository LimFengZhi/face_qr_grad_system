import os
import pickle
import numpy as np
from deepface import DeepFace

class FaceRecognitionDeepFace:
    
    def __init__(self, file_path, threshold=0.4, 
                 model_name='Facenet512', 
                 detector_backend='retinaface'):
        self.file_path = file_path
        self.threshold = threshold
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.known_encodings = []
        self.known_ids = []
        
        print(f"Loading {model_name} model...")
        try:
            # Updated warm-up for newer DeepFace versions
            dummy_img = np.zeros((160, 160, 3), dtype=np.uint8)
            DeepFace.represent(
                dummy_img, 
                model_name=model_name, 
                detector_backend='skip',  # Skip detection for dummy image
                enforce_detection=False
            )
            print(f"✓ DeepFace initialized: {model_name} + {detector_backend}")
        except Exception as e:
            # Try alternative warm-up method
            try:
                from deepface.basemodels import Facenet512
                print(f"✓ DeepFace initialized (alt): {model_name}")
            except:
                print(f"⚠ Model warm-up skipped: {e}")
        
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
        """Compute cosine distance"""
        enc1, enc2 = np.array(enc1), np.array(enc2)
        dot = np.dot(enc1, enc2)
        norm1, norm2 = np.linalg.norm(enc1), np.linalg.norm(enc2)
        if norm1 == 0 or norm2 == 0:
            return 1.0
        return 1 - (dot / (norm1 * norm2))
    
    def detect_face(self, face_img):
        """
        Detect face in image
        
        Returns:
            face_region: Dict with facial_area info, or None if no face
        """
        try:
            faces = DeepFace.extract_faces(
                face_img,
                detector_backend=self.detector_backend,
                enforce_detection=True,
                align=False
            )
            
            if len(faces) == 0:
                print(" No face detected")
                return None
            
            largest_face = max(faces, key=lambda f: f['facial_area']['w'] * f['facial_area']['h'])
            return largest_face
            
        except ValueError as e:
            if "Face could not be detected" in str(e):
                print(" No face detected")
                return None
            print(f" Detection error: {e}")
            return None
        except Exception as e:
            print(f" Detection error: {e}")
            return None
    
    def recognise_face(self, face_img, face_region=None):
        """
        Get face embedding
        
        Args:
            face_img: Input image (RGB)
            face_region: Optional face region (not used, DeepFace does detection)
        
        Returns:
            embedding: Face embedding vector
            success: Boolean
        """
        try:
            result = DeepFace.represent(
                face_img,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=True
            )
            
            if not result or len(result) == 0:
                print(" Failed")
                return None, False
            
            embedding = np.array(result[0]['embedding'])
            print(" Success")
            return embedding, True
            
        except ValueError as e:
            if "Face could not be detected" in str(e):
                print(" Failed")
                return None, False
            print(f" Failed: {e}")
            return None, False
        except Exception as e:
            print(f" Failed: {e}")
            return None, False
    
    def compare_encoding(self, encoding):
        """Compare encoding against known faces"""
        if len(self.known_encodings) == 0:
            return None, None, False
        
        best_dist = float('inf')
        best_id = None
        
        for known_enc, known_id in zip(self.known_encodings, self.known_ids):
            dist = self.compute_distance(encoding, known_enc)
            if dist < best_dist:
                best_dist = dist
                best_id = known_id
        
        if best_dist < self.threshold:
            return best_id, float(best_dist), True
        else:
            return None, float(best_dist), False
    
    def register_face(self, encoding, person_id):
        """Register encoding"""
        self.known_encodings.append(encoding)
        self.known_ids.append(person_id)
        self.save_encoding()
    
    def clear_encodings(self):
        self.known_encodings = []
        self.known_ids = []
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
        print("✓ Encodings cleared")