import cv2 as cv
import threading
import time
from queue import Queue

class Camera:

    def __init__(self, camera_id=0, width=640, height=480, fps=30):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        
        self.cap = None
        self.is_running = False
        self.frame = None
        self.frame_count = 0
        
        # Threading
        self.thread = None
        self.lock = threading.Lock()
        
    def start(self):
        if self.is_running:
            print("Camera already running")
            return False
        
        # Open camera
        self.cap = cv.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"Error: Cannot open camera {self.camera_id}")
            return False
        
        # Set camera properties
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv.CAP_PROP_FPS, self.fps)
        
        # Read actual properties (might differ from requested)
        actual_width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv.CAP_PROP_FPS))
        
        print(f"Camera {self.camera_id} started")
        print(f"   Resolution: {actual_width}x{actual_height}")
        print(f"   FPS: {actual_fps}")
        
        self.is_running = True
        
        # Start background thread
        self.thread = threading.Thread(target=self._update_frame, daemon=True)
        self.thread.start()
        
        return True
    
    def _update_frame(self):
        while self.is_running:
            ret, frame = self.cap.read()
            
            if ret:
                with self.lock:
                    self.frame = frame
                    self.frame_count += 1
            else:
                print("⚠️  Warning: Failed to read frame")
                time.sleep(0.01)
    
    def read(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()
    
    def get_frame_count(self):
        return self.frame_count
    
    def is_opened(self):
        return self.is_running and self.cap is not None and self.cap.isOpened()
    
    def stop(self):
        if not self.is_running:
            print("Camera not running")
            return
        
        self.is_running = False
        
        # Wait for thread to finish
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        
        # Release camera
        if self.cap is not None:
            self.cap.release()
        
        print("Camera stopped")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    def __del__(self):
        self.stop()