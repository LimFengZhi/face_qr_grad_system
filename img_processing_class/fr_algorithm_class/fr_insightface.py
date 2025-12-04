import insightface
import numpy as np
import os
import pickle

class FaceRecognitionInsightFace:
    
    def __init__(self, file_path, threshold=0.4, model_name='buffalo_s', 
                 ctx_id=-1, det_size=(640, 640)):
        self.file_path = file_path
        self.threshold = threshold
        self.model_name = model_name
        self.ctx_id = ctx_id
        self.det_size = det_size
        self.known_encodings = []
        self.known_ids = []
        
        print(f"Loading InsightFace model ({model_name})...")
        try:
            self.model = insightface.app.FaceAnalysis(name=model_name)
            self.model.prepare(ctx_id=ctx_id, det_size=det_size)
            print(f"✓ InsightFace initialized: {model_name} (det_size={det_size})")
        except Exception as e:
            print(f"Model initialization error: {e}")
            self.model = None
        
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
        Detect face in image 
        
        Args:
            face_img: Input image (BGR)
        
        Returns:
            face_region: InsightFace Face object, or None if no face
        """
        try:
            if self.model is None:
                print(" Model not initialized")
                return None
            
            faces = self.model.get(face_img)
            
            if len(faces) == 0:
                print(" No face detected")
                return None
            
            # Return largest face (by area)
            largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            return largest_face
            
        except Exception as e:
            print(f" Detection error: {e}")
            return None
    
    def recognise_face(self, face_img, face_region=None):
        """
        Extract face encoding 
        
        Args:
            face_img: Input image (BGR)
            face_region: Face object from detect_face (contains embedding)
        
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
            
            # InsightFace already computes embedding during detection
            embedding = face_region.normed_embedding
            
            if embedding is None:
                print(" Failed")
                return None, False
            
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