import cv2 as cv
import numpy as np
import face_recognition

class AdaptivePreprocessor:
    """Adaptive preprocessing based on image quality analysis"""
    
    def __init__(self):
        pass
    
    def estimate_brightness(self, image):
        """Estimate image brightness (0-255)"""
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        return np.mean(gray)
    
    def estimate_contrast(self, image):
        """Estimate image contrast"""
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        return np.std(gray)
    
    def estimate_blur(self, image):
        """Estimate blur using Laplacian variance (lower = more blurry)"""
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        return cv.Laplacian(gray, cv.CV_64F).var()
    
    def estimate_noise(self, image):
        """Estimate noise level using median filter difference"""
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        median = cv.medianBlur(gray, 5)
        diff = cv.absdiff(gray, median)
        return np.mean(diff)
    
    def apply_clahe(self, image, clip_limit=2.0):
        """CLAHE for illumination normalization"""
        lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
        l, a, b = cv.split(lab)
        clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        return cv.cvtColor(cv.merge((cl, a, b)), cv.COLOR_LAB2BGR)
    
    def apply_gamma_correction(self, image, gamma):
        """
        Gamma correction for brightness adjustment
        gamma < 1.0 = brighten image
        gamma > 1.0 = darken image
        """
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(256)]).astype("uint8")
        return cv.LUT(image, table)
    
    def apply_unsharp_mask(self, image, sigma=1.0, strength=0.5):
        """Unsharp mask for mild sharpening"""
        blurred = cv.GaussianBlur(image, (0, 0), sigma)
        sharpened = cv.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
        return sharpened
    
    def apply_bilateral_denoise(self, image):
        """Bilateral filter - preserves edges while denoising"""
        return cv.bilateralFilter(image, 5, 50, 50)
    
    def normalize_color(self, image):
        """Normalize color distribution using histogram equalization on YCrCb"""
        ycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
        y, cr, cb = cv.split(ycrcb)
        y_eq = cv.equalizeHist(y)
        return cv.cvtColor(cv.merge((y_eq, cr, cb)), cv.COLOR_YCrCb2BGR)
    
    def process(self, image):
        """Adaptive preprocessing pipeline"""
        processed = image.copy()
        
        brightness = self.estimate_brightness(processed)
        contrast = self.estimate_contrast(processed)
        blur_score = self.estimate_blur(processed)
        noise_level = self.estimate_noise(processed)
        
        # 1. Fix low light (brightness < 80)
        if brightness < 80:
            gamma = 0.6  # Brighten
            gamma = 0.5 + (brightness / 160)  # Range: 0.5-1.0 based on darkness
            processed = self.apply_gamma_correction(processed, gamma)
        
        # Fix overexposure (brightness > 180)
        elif brightness > 180:
            gamma = 1.5  # Darken
            processed = self.apply_gamma_correction(processed, gamma)
        
        # 2. Fix low contrast (std < 40)
        if contrast < 35:
            processed = self.apply_clahe(processed, clip_limit=3.0)
        elif contrast < 55:
            processed = self.apply_clahe(processed, clip_limit=2.0)
        
        # 3. Fix blur (variance < 100 indicates blur)
        if 80 < blur_score < 300:
            processed = self.apply_unsharp_mask(processed, sigma=1.0, strength=0.8)
        elif 300 <= blur_score < 500:
            processed = self.apply_unsharp_mask(processed, sigma=1.5, strength=0.5)
        
        
        if noise_level > 12:  # Only denoise noisy images
            processed = self.apply_bilateral_denoise(processed)
        
        return processed
    

    def align_face(self, image, face_location):
        """
        Align face by rotating based on eye positions
        
        Args:
            image: Input image
            face_location: Single face location tuple (top, right, bottom, left)
        
        Returns:
            aligned_face: Rotated and aligned face
        """
        # Handle if list is passed instead of tuple
        if isinstance(face_location, list):
            if len(face_location) == 0:
                return image
            face_location = face_location[0]
        
        # Validate face_location is a tuple with 4 elements
        if not isinstance(face_location, tuple) or len(face_location) != 4:
            return image
        
        try:
            face_landmarks = face_recognition.face_landmarks(image, [face_location])
            
            if len(face_landmarks) > 0:
                landmarks = face_landmarks[0]
                
                # Check if eyes exist in landmarks
                if 'left_eye' not in landmarks or 'right_eye' not in landmarks:
                    return image
                
                left_eye = landmarks['left_eye']
                right_eye = landmarks['right_eye']
                
                left_eye_center = np.mean(left_eye, axis=0).astype(int)
                right_eye_center = np.mean(right_eye, axis=0).astype(int)
                
                dy = right_eye_center[1] - left_eye_center[1]
                dx = right_eye_center[0] - left_eye_center[0]
                angle = np.degrees(np.arctan2(dy, dx))
                
                top, right, bottom, left = face_location
                center = ((left + right) // 2, (top + bottom) // 2)
                
                rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)
                aligned_image = cv.warpAffine(image, rotation_matrix, 
                                             (image.shape[1], image.shape[0]),
                                             flags=cv.INTER_CUBIC)
                
                return aligned_image
        except Exception as e:
            print(f"Alignment error: {e}")
        
        return image
    
    @staticmethod
    def get_largest_face(face_locations):
        """Get the largest face from list of face locations"""
        if not face_locations:
            return []
        return [max(face_locations, key=lambda f: (f[2] - f[0]) * (f[1] - f[3]))]