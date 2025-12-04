import cv2 as cv

def draw_detections(frame, face_id, face_box, qr_id, qr_rect):
    """Draw bounding boxes"""
    if face_box is not None:
        x1, y1, x2, y2 = int(face_box[0]), int(face_box[1]), int(face_box[2]), int(face_box[3])
        color = (0, 255, 0) if face_id else (0, 165, 255)
        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"Face: {face_id}" if face_id else "Face: Unknown"
        cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    if qr_rect is not None:
        x, y, w, h = qr_rect
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
        label = f"QR: {qr_id}" if qr_id else "QR"
        cv.putText(frame, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    return frame