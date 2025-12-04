import cv2 as cv
import face_recognition
import numpy as np
import os
import pickle


class FaceRecognitionHogDlib:
    def __init__(self, file_path, confidence, model='hog'):
        self.known_list_encoding = []
        self.known_list_ids = []
        self.file_path = file_path
        self.confidence = confidence
        self.detect_model = model
        self.load_encodings()
    
    def load_encodings(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'rb') as f:
                data = pickle.load(f)
                self.known_list_encoding = data.get('encodings', [])
                self.known_list_ids = data.get('ids', [])
        else:
            self.known_list_encoding = []
            self.known_list_ids = []


    def save_encoding(self):
        data = {
            'encodings': self.known_list_encoding,
            'ids': self.known_list_ids
        }
        with open(self.file_path, 'wb') as f:
            pickle.dump(data, f)

    def compare_encoding(self, encoding):
        if len(self.known_list_encoding) == 0:
            return None, None, False
        
        matches = face_recognition.compare_faces(self.known_list_encoding, encoding, tolerance=self.confidence)
        distances = face_recognition.face_distance(self.known_list_encoding, encoding)
        
        if True in matches:
            best_match_index = int(np.argmin(distances))
            best_distance = float(distances[best_match_index])  # Convert to Python float
            return self.known_list_ids[best_match_index], best_distance, True
        else:
            best_distance = float(np.min(distances)) if len(distances) > 0 else None
            return None, best_distance, False
        
    def recognise_face(self, face_img, face_location):
        if isinstance(face_location, list):
            known_locations = face_location
        else:
            known_locations = [face_location]
        
        face_encoding = face_recognition.face_encodings(face_img, model='small', known_face_locations=known_locations)

        if len(face_encoding) > 0:
            successful = True
            print(" Success")
            return face_encoding[0], successful
        else:
            print(" Failed")
            successful = False
            return None, successful
    
    def detect_face(self, face_img):
        face_loc = face_recognition.face_locations(face_img, model=self.detect_model)
        if not face_loc:
            print(" No face detected")
            return None
        else:
            return face_loc


    def register_face(self, encoding, id):
        self.known_list_encoding.append(encoding)
        self.known_list_ids.append(id)
        self.save_encoding()
        