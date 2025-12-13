from img_processing_class.utility_class.qr_code_scanner import QRCodeScanner
import cv2 as cv
import time

# Create scanner
scanner = QRCodeScanner()

# Performance settings
SKIP_FRAMES = 3  # Only scan every 3rd frame for smoother performance
last_qr_data = None
last_qr_time = 0
QR_DISPLAY_TIME = 2.0  # Show QR data for 2 seconds

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)   # Lower resolution for speed
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv.CAP_PROP_FPS, 30)

frame_count = 0
fps_time = time.time()
fps_count = 0
fps = 0

print("QR Scanner - Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    fps_count += 1
    
    # Calculate FPS
    if time.time() - fps_time >= 1:
        fps = fps_count
        fps_count = 0
        fps_time = time.time()
    
    current_time = time.time()
    
    # Only scan every Nth frame for smooth performance
    if frame_count % SKIP_FRAMES == 0:
        try:
            # Quick scan only (no heavy preprocessing)
            from pyzbar import pyzbar
            decoded = pyzbar.decode(frame)
            
            if decoded:
                last_qr_data = decoded[0].data.decode('utf-8')
                last_qr_time = current_time
                
                # Draw rectangle around QR
                rect = decoded[0].rect
                cv.rectangle(frame, 
                            (rect.left, rect.top), 
                            (rect.left + rect.width, rect.top + rect.height), 
                            (0, 255, 0), 3)
        except:
            pass
    
    # Display last detected QR for a few seconds
    if last_qr_data and (current_time - last_qr_time) < QR_DISPLAY_TIME:
        # QR detected recently - show green
        cv.putText(frame, f"QR: {last_qr_data}", (10, 40), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv.putText(frame, "DETECTED", (10, 70), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        # Scanning - show yellow
        cv.putText(frame, "Scanning...", (10, 40), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Show FPS
    cv.putText(frame, f"FPS: {fps}", (10, frame.shape[0] - 10), 
              cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Display
    cv.imshow('QR Scanner - Smooth', frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

