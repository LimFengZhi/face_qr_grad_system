
import threading
import cv2 as cv

class Camera:
    def __init__(self):
        self.cap = None
        self.lock = threading.Lock()
        self.running = False
        
    def start(self):
        with self.lock:
            if self.cap is None or not self.cap.isOpened():
                for idx in [0, 1]:
                    try:
                        self.cap = cv.VideoCapture(idx, cv.CAP_DSHOW)
                        if self.cap.isOpened():
                            ret, frame = self.cap.read()
                            if ret:
                                self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
                                self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
                                self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
                                print(f"✓ Camera {idx} opened")
                                self.running = True
                                return True
                        self.cap.release()
                    except:
                        pass
                return False
            return True
    
    def stop(self):
        with self.lock:
            self.running = False
            if self.cap:
                self.cap.release()
                self.cap = None
    
    def read(self):
        with self.lock:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    return frame
        return None