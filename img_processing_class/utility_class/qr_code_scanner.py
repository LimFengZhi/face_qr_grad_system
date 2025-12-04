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
    
    def enhance_image(self, image, scale=2.0):
        """Enhance and upscale image for better QR detection"""
        # Upscale
        h, w = image.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)
        upscaled = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_CUBIC)
        
        # Sharpen
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv.filter2D(upscaled, -1, kernel)
        
        return sharpened
    
    def find_qr_region(self, image):
        """
        Find potential QR code regions using contour detection
        
        Returns:
            List of (x, y, w, h) bounding boxes for potential QR regions
        """
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv.Canny(gray, 50, 150)
        
        # Dilate to connect edges
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        regions = []
        h, w = image.shape[:2]
        min_size = min(w, h) * 0.05  # Minimum 5% of image
        max_size = min(w, h) * 0.8   # Maximum 80% of image
        
        for contour in contours:
            x, y, cw, ch = cv.boundingRect(contour)
            
            # Filter by size and aspect ratio (QR codes are roughly square)
            if min_size < cw < max_size and min_size < ch < max_size:
                aspect_ratio = cw / ch if ch > 0 else 0
                if 0.5 < aspect_ratio < 2.0:  # Roughly square
                    # Add padding around the region
                    padding = int(max(cw, ch) * 0.2)
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    cw = min(w - x, cw + 2 * padding)
                    ch = min(h - y, ch + 2 * padding)
                    regions.append((x, y, cw, ch))
        
        # Sort by area (largest first)
        regions.sort(key=lambda r: r[2] * r[3], reverse=True)
        
        return regions[:5]  # Return top 5 candidates
    
    def scan_with_zoom(self, image, zoom_scales=[1.0, 1.5, 2.0, 3.0]):
        """
        Scan QR with multiple zoom levels
        
        Args:
            image: BGR image
            zoom_scales: List of scale factors to try
        
        Returns:
            List of decoded QR data dicts
        """
        results = []
        
        for scale in zoom_scales:
            if scale == 1.0:
                scan_img = image
            else:
                scan_img = self.enhance_image(image, scale)
            
            # Try pyzbar
            decoded = pyzbar.decode(scan_img)
            if decoded:
                for obj in decoded:
                    # Adjust rect coordinates back to original scale
                    rect = obj.rect
                    if scale != 1.0:
                        adjusted_rect = pyzbar.Rect(
                            int(rect.left / scale),
                            int(rect.top / scale),
                            int(rect.width / scale),
                            int(rect.height / scale)
                        )
                    else:
                        adjusted_rect = rect
                    
                    results.append({
                        'data': obj.data.decode('utf-8'),
                        'type': obj.type,
                        'rect': adjusted_rect
                    })
                return results  # Found at this scale, return
            
            # Try preprocessed
            preprocessed = self.preprocess(scan_img)
            decoded = pyzbar.decode(preprocessed)
            if decoded:
                for obj in decoded:
                    rect = obj.rect
                    if scale != 1.0:
                        adjusted_rect = pyzbar.Rect(
                            int(rect.left / scale),
                            int(rect.top / scale),
                            int(rect.width / scale),
                            int(rect.height / scale)
                        )
                    else:
                        adjusted_rect = rect
                    
                    results.append({
                        'data': obj.data.decode('utf-8'),
                        'type': obj.type,
                        'rect': adjusted_rect
                    })
                return results
        
        return results
    
    def scan_regions(self, image):
        """
        Find QR regions, crop and zoom each one for scanning
        
        Args:
            image: BGR image
        
        Returns:
            List of decoded QR data dicts
        """
        # First try full image
        results = self.scan_with_zoom(image, [1.0, 1.5, 2.0])
        if results:
            return results
        
        # Find potential QR regions
        regions = self.find_qr_region(image)
        
        for (x, y, w, h) in regions:
            # Crop the region
            cropped = image[y:y+h, x:x+w]
            
            if cropped.size == 0:
                continue
            
            # Try scanning the cropped region with zoom
            decoded = self.scan_with_zoom(cropped, [2.0, 3.0, 4.0])
            
            if decoded:
                # Adjust coordinates back to original image
                for item in decoded:
                    rect = item['rect']
                    item['rect'] = pyzbar.Rect(
                        x + rect.left,
                        y + rect.top,
                        rect.width,
                        rect.height
                    )
                return decoded
        
        return []
    
    # ==================== SCANNING ====================
    
    def scan(self, image):
        """
        Scan QR code from image with auto-zoom for small QR codes
        
        Args:
            image: BGR image (numpy array)
        
        Returns:
            List of decoded QR data dicts
        """
        # Quick scan first (fastest)
        decoded = pyzbar.decode(image)
        if decoded:
            results = []
            for obj in decoded:
                results.append({
                    'data': obj.data.decode('utf-8'),
                    'type': obj.type,
                    'rect': obj.rect
                })
            return results
        
        # Try preprocessed
        preprocessed = self.preprocess(image)
        decoded = pyzbar.decode(preprocessed)
        if decoded:
            results = []
            for obj in decoded:
                results.append({
                    'data': obj.data.decode('utf-8'),
                    'type': obj.type,
                    'rect': obj.rect
                })
            return results
        
        # Try with zoom and region detection for small QR codes
        return self.scan_regions(image)
    
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