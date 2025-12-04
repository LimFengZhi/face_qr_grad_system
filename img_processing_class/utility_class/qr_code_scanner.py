import cv2 as cv
import numpy as np
from pyzbar import pyzbar
import qrcode

class QRCodeScanner:
    """QR Code Scanner and Generator for images"""
    
    def __init__(self):
        self.cv_detector = cv.QRCodeDetector()
    
    # ==================== GENERATION ====================
    
    def generate(self, data, size=300, border=4, error_correction='M'):
        """
        Generate QR code image
        
        Args:
            data: String data to encode
            size: Output image size in pixels
            border: Border size (minimum 4)
            error_correction: 'L' (7%), 'M' (15%), 'Q' (25%), 'H' (30%)
        
        Returns:
            QR code as numpy array (BGR)
        """
        # Error correction levels
        ec_levels = {
            'L': qrcode.constants.ERROR_CORRECT_L,
            'M': qrcode.constants.ERROR_CORRECT_M,
            'Q': qrcode.constants.ERROR_CORRECT_Q,
            'H': qrcode.constants.ERROR_CORRECT_H
        }
        
        qr = qrcode.QRCode(
            version=1,
            error_correction=ec_levels.get(error_correction, qrcode.constants.ERROR_CORRECT_M),
            box_size=10,
            border=border
        )
        qr.add_data(data)
        qr.make(fit=True)
        
        # Create PIL image and convert to numpy
        pil_img = qr.make_image(fill_color="black", back_color="white")
        img = np.array(pil_img.convert('RGB'))
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        
        # Resize to desired size
        img = cv.resize(img, (size, size), interpolation=cv.INTER_NEAREST)
        
        return img
    
    def generate_and_save(self, data, output_path, size=300, border=4):
        """
        Generate QR code and save to file
        
        Args:
            data: String data to encode
            output_path: Path to save image
            size: Output image size in pixels
            border: Border size
        
        Returns:
            True if successful
        """
        img = self.generate(data, size, border)
        success = cv.imwrite(output_path, img)
        
        if success:
            print(f"✓ QR code saved: {output_path}")
        else:
            print(f"✗ Failed to save: {output_path}")
        
        return success
    
    # ==================== PREPROCESSING ====================
    
    def preprocess(self, frame):
        """Preprocess image for better QR detection"""
        # 1. Convert to Grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # 2. Apply Gaussian Blur
        blurred = cv.GaussianBlur(gray, (5, 5), 0)
        
        # 3. Adaptive Thresholding
        binary = cv.adaptiveThreshold(
            blurred, 255,
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY, 11, 2
        )
        return binary
    
    # ==================== SCANNING ====================
    
    def scan(self, image):
        """
        Scan QR code from image
        
        Args:
            image: BGR image (numpy array)
        
        Returns:
            List of decoded QR data dicts
        """
        results = []
        
        # Try pyzbar on original
        decoded = pyzbar.decode(image)
        for obj in decoded:
            results.append({
                'data': obj.data.decode('utf-8'),
                'type': obj.type,
                'rect': obj.rect
            })
        
        if results:
            return results
        
        # Try preprocessed image
        preprocessed = self.preprocess(image)
        decoded = pyzbar.decode(preprocessed)
        for obj in decoded:
            results.append({
                'data': obj.data.decode('utf-8'),
                'type': obj.type,
                'rect': obj.rect
            })
        
        return results
    
    def scan_file(self, image_path):
        """
        Scan QR code from file path
        
        Args:
            image_path: Path to image file
        
        Returns:
            List of decoded QR data, or empty list
        """
        image = cv.imread(image_path)
        if image is None:
            print(f"Error: Cannot read {image_path}")
            return []
        
        results = self.scan(image)
        
        for qr in results:
            print(f"✓ Found: {qr['data']}")
        
        if not results:
            print("No QR code found")
        
        return results