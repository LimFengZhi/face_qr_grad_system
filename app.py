from flask import Flask, render_template, Response, jsonify, request
import cv2 as cv
import threading
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from img_processing_class.utility_class.adaptive_preprocessor import AdaptivePreprocessor
from img_processing_class.utility_class.qr_code_scanner import QRCodeScanner
from img_processing_class.utility_class.box_helper import draw_detections
from img_processing_class.utility_class.text_to_speach import TextToSpeech
from img_processing_class.utility_class.student_database import init_database, graduation_level
from img_processing_class.utility_class.camera import Camera
from img_processing_class.fr_algorithm_class.fr_insightface import FaceRecognitionInsightFace
from img_processing_class.fr_algorithm_class.fr_hog_dlib import FaceRecognitionHogDlib
from img_processing_class.fr_algorithm_class.fr_deepface import FaceRecognitionDeepFace
from img_processing_class.utility_class.email_sender import get_email_sender
from img_processing_class.utility_class.person_tracker import PersonTracker

app = Flask(__name__)

# ==================== GLOBAL STATE ====================
class AppState:
    def __init__(self):
        self.queue_started = False
        self.current_queue_index = 0
        self.verification_state = 'waiting'
        self.display_start_time = 0
        self.queue_list = []
        self.current_algorithm = "HOG + Dlib"
        self.tts_enabled = True
        self.display_duration = 8
        self.last_face_id = None
        self.last_face_box = None
        self.last_face_detected = False
        self.last_qr_id = None
        self.last_qr_rect = None
        self.last_qr_detected = False
        self.verified_student = None
        self.email_enabled = False
        self.person_tracking_enabled = True
        self.lock = threading.Lock()

state = AppState()

# ==================== LOAD MODELS ====================
print("🚀 Loading models...")
executor = ThreadPoolExecutor(max_workers=4)

def load_hog_dlib():
    try:

        model = FaceRecognitionHogDlib(file_path="data/encodings/hb_encoding.pkl", confidence=0.6)
        return ("HOG + Dlib", model)
    except Exception as e:
        print(f"HOG + Dlib failed: {e}")
        return ("HOG + Dlib", None)

def load_deepface():
    try:
        model = FaceRecognitionDeepFace(
            file_path="data/encodings/deepface_facenet512.pkl",
            threshold=0.68, model_name='Facenet512', detector_backend='retinaface'
        )
        return ("DeepFace", model)
    except Exception as e:
        print(f"DeepFace failed: {e}")
        return ("DeepFace", None)

def load_insightface():
    try:
        model = FaceRecognitionInsightFace(
            file_path="data/encodings/insightface_buffalo.pkl",
            threshold=0.3, model_name='buffalo_s', ctx_id=-1
        )
        return ("InsightFace", model)
    except Exception as e:
        print(f"InsightFace failed: {e}")
        return ("InsightFace", None)

def load_mtcnn_facenet():
    try:
        from img_processing_class.fr_algorithm_class.fr_mtcnn_facenet import FaceRecognitionMTCNNFaceNet
        model = FaceRecognitionMTCNNFaceNet(file_path="data/encodings/mtcnn_facenet.pkl", threshold=0.4)
        return ("MTCNN + FaceNet", model)
    except Exception as e:
        print(f"MTCNN + FaceNet failed: {e}")
        return ("MTCNN + FaceNet", None)

def load_person_tracker():
    """Load YOLOv8 person tracker"""
    try:
        tracker = PersonTracker(
            model_name='yolov8n.pt',
            confidence=0.5,
            verification_zone=(0.3, 0.1, 0.7, 0.90),  # Center zone
            max_tracks=1      # Only track 1 person at a time
        )
        return tracker
    except Exception as e:
        print(f"Person Tracker failed: {e}")
        return None

# Load models in parallel (except MTCNN)
all_models = {}
futures = [
    executor.submit(load_hog_dlib),
    executor.submit(load_deepface),
    executor.submit(load_insightface),
]
for future in futures:
    name, model = future.result()
    all_models[name] = model
    if model:
        print(f"✓ {name} loaded")

name, model = load_mtcnn_facenet()
all_models[name] = model
if model:
    print(f"✓ {name} loaded")

# Load person tracker
person_tracker = load_person_tracker()

preprocessor = AdaptivePreprocessor()
qr_scanner = QRCodeScanner()
tts = TextToSpeech()
db = init_database("data/student_list.csv", "data/students.db")

available_algorithms = [name for name, model in all_models.items() if model is not None]
print(f"✅ Ready! Available: {available_algorithms}")

# ==================== DETECTION FUNCTIONS ====================
def detect_face_pipeline(frame, model, algo_name):
    if model is None:
        return None, None, False
    try:
        processed = preprocessor.process(frame)
        
        if algo_name == "InsightFace":
            face_region = model.detect_face(processed)
            if face_region is None:
                return None, None, False
            bbox = face_region.bbox.astype(int)
            box = (bbox[0], bbox[1], bbox[2], bbox[3])
            encoding, success = model.recognise_face(processed, face_region)
            if not success:
                return None, box, True
            matched_id, distance, matched = model.compare_encoding(encoding)
            return matched_id if matched else None, box, True
        
        elif algo_name == "MTCNN + FaceNet":
            rgb_frame = cv.cvtColor(processed, cv.COLOR_BGR2RGB)
            face_region = model.detect_face(rgb_frame)
            if face_region is None:
                return None, None, False
            x, y, w, h = face_region['box']
            x, y = abs(x), abs(y)
            box = (x, y, x + w, y + h)
            encoding, success = model.recognise_face(rgb_frame, face_region)
            if not success:
                return None, box, True
            matched_id, distance, matched = model.compare_encoding(encoding)
            return matched_id if matched else None, box, True
        
        elif algo_name == "DeepFace":
            rgb_frame = cv.cvtColor(processed, cv.COLOR_BGR2RGB)
            face_region = model.detect_face(rgb_frame)
            if face_region is None:
                return None, None, False
            # DeepFace returns dict with 'facial_area': {'x', 'y', 'w', 'h'}
            fa = face_region['facial_area']
            x, y, w, h = fa['x'], fa['y'], fa['w'], fa['h']
            box = (x, y, x + w, y + h)
            encoding, success = model.recognise_face(rgb_frame, face_region)
            if not success:
                return None, box, True
            matched_id, distance, matched = model.compare_encoding(encoding)
            return matched_id if matched else None, box, True
        
        else:  # HOG + Dlib
            rgb_frame = cv.cvtColor(processed, cv.COLOR_BGR2RGB)
            face_loc = model.detect_face(rgb_frame)
            if face_loc is None or len(face_loc) == 0:
                return None, None, False
            if isinstance(face_loc, list) and len(face_loc) > 0:
                face_loc = max(face_loc, key=lambda f: (f[2] - f[0]) * (f[1] - f[3]))
            top, right, bottom, left = face_loc
            box = (left, top, right, bottom)
            encoding, success = model.recognise_face(
                rgb_frame, [face_loc] if isinstance(face_loc, tuple) else face_loc
            )
            if not success:
                return None, box, True
            matched_id, distance, matched = model.compare_encoding(encoding)
            return matched_id if matched else None, box, True
    except Exception as e:
        print(f"Detection error ({algo_name}): {e}")
        return None, None, False

def detect_qr_pipeline(frame):
    try:
        qr_results = qr_scanner.scan(frame)
        if qr_results:
            qr_id = qr_results[0]['data']
            rect = qr_results[0]['rect']
            qr_rect = (rect.left, rect.top, rect.width, rect.height)
            return qr_id, qr_rect, True
    except:
        pass
    return None, None, False

def find_student_image(student_id):
    for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']:
        img_path = f"data/Image/{student_id}{ext}"
        if os.path.exists(img_path):
            return img_path
    return None



camera = Camera()

def generate_frames():
    global state, person_tracker
    frame_count = 0
    fps_time = time.time()
    fps_count = 0
    fps = 0
    
    # Create executor for parallel detection
    detection_executor = ThreadPoolExecutor(max_workers=2)
    
    while state.queue_started:
        frame = camera.read()
        if frame is None:
            time.sleep(0.03)
            continue
        
        frame_count += 1
        fps_count += 1
        
        if time.time() - fps_time >= 1:
            fps = fps_count
            fps_count = 0
            fps_time = time.time()
        
        now = time.time()
        
        # Person tracking detection (runs every frame for smooth tracking)
        person_in_zone = None
        if state.person_tracking_enabled and person_tracker is not None:
            person_in_zone = person_tracker.get_person_in_zone(frame)
        
        with state.lock:
            idx = state.current_queue_index
            if idx >= len(state.queue_list):
                state.queue_started = False
                break
            
            student = state.queue_list[idx]
            algo = state.current_algorithm
            model = all_models.get(algo)
            
            # Only check attended status when in waiting state (not during display countdown)
            if state.verification_state == 'waiting':
                # Check if student already attended (sync with database)
                db_student = db.get_student(student['student_id'])
                if db_student and db_student.get('attended', False):
                    # Skip this student, they already attended
                    state.queue_list[idx]['attended'] = True
                    state.current_queue_index += 1
                    state.last_face_id = None
                    state.last_face_box = None
                    state.last_face_detected = False
                    state.last_qr_id = None
                    state.last_qr_rect = None
                    state.last_qr_detected = False
                    # Check if queue finished
                    if state.current_queue_index >= len(state.queue_list):
                        state.queue_started = False
                    # Still render this frame, will get next student on next iteration
                    if state.person_tracking_enabled and person_tracker:
                        disp = person_tracker.draw_tracking(frame)
                    else:
                        disp = draw_detections(frame, None, None, None, None)
                    cv.putText(disp, f"FPS:{fps} | {algo}", (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                    cv.putText(disp, "Skipping attended student...", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
                else:
                    # Normal detection - only run if person is in zone (or tracking disabled)
                    expected_id = student['student_id']
                    
                    # Check if we should run face/QR detection
                    should_detect = True
                    if state.person_tracking_enabled and person_tracker:
                        # Only detect if person is in verification zone
                        should_detect = person_in_zone is not None

                    if should_detect and frame_count % 30 == 0:
                        # Run face and QR detection
                        face_future = detection_executor.submit(detect_face_pipeline, frame.copy(), model, algo)
                        qr_future = detection_executor.submit(detect_qr_pipeline, frame.copy())
                        
                        # Wait for both to complete
                        state.last_face_id, state.last_face_box, state.last_face_detected = face_future.result()
                        state.last_qr_id, state.last_qr_rect, state.last_qr_detected = qr_future.result()
                    elif not should_detect:
                        # Reset when no person in zone
                        state.last_face_id = None
                        state.last_face_box = None
                        state.last_face_detected = False
                        state.last_qr_id = None
                        state.last_qr_rect = None
                        state.last_qr_detected = False
                    
                    if (state.last_face_id and state.last_qr_id and 
                        str(state.last_face_id) == str(state.last_qr_id) == str(expected_id)):
                        state.verification_state = 'displaying'
                        state.display_start_time = now
                        state.verified_student = student
                        db.mark_attended(expected_id)
                        state.queue_list[idx]['attended'] = True
                        if state.tts_enabled:
                            grad_level = graduation_level(student.get('cgpa', 0) or 0)
                            announcement = f"Congratulations {student['name']}, graduated with {grad_level}"
                            threading.Thread(target=tts.speak, args=(announcement,), daemon=True).start()
                    
                    # Draw detections with person tracking overlay
                    if state.person_tracking_enabled and person_tracker:
                        disp = person_tracker.draw_tracking(frame)
                        # Overlay face/QR detections on top
                        if state.last_face_box:
                            x1, y1, x2, y2 = state.last_face_box
                            # Green if matches expected ID, Red if wrong person detected
                            face_match = state.last_face_id and str(state.last_face_id) == str(expected_id)
                            if face_match:
                                color = (0, 255, 0)  # Green - correct person
                                label = f"Face: {state.last_face_id}"
                            elif state.last_face_id:
                                color = (0, 0, 255)  # Red - wrong person
                                label = f"Face: {state.last_face_id}"
                            else:
                                color = (0, 165, 255)  # Orange - face detected but no match
                                label = "Face: Unknown"
                            cv.rectangle(disp, (x1, y1), (x2, y2), color, 2)
                            cv.putText(disp, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        if state.last_qr_rect:
                            x, y, w, h = state.last_qr_rect
                            # Green if matches expected ID, Red if wrong QR
                            qr_match = state.last_qr_id and str(state.last_qr_id) == str(expected_id)
                            if qr_match:
                                color = (0, 255, 0)  # Green - correct QR
                                label = f"QR: {state.last_qr_id}"
                            elif state.last_qr_id:
                                color = (0, 0, 255)  # Red - wrong QR
                                label = f"QR: {state.last_qr_id}"
                            else:
                                color = (0, 165, 255)  # Orange - QR detected but no data
                                label = "QR: None"
                            cv.rectangle(disp, (x, y), (x + w, y + h), color, 2)
                            cv.putText(disp, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    else:
                        disp = draw_detections(frame, state.last_face_id, state.last_face_box, 
                                               state.last_qr_id, state.last_qr_rect)
                    
                    cv.putText(disp, f"FPS:{fps} | {algo}", (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                    
                     # Show waiting status with detailed feedback
                    if state.person_tracking_enabled and person_tracker:
                        if person_in_zone:
                            # Person is in zone - show detailed status
                            status_msg = f"Wait: {student['name']}"
                            cv.putText(disp, status_msg, (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                            
                            # Check face and QR status
                            face_match = state.last_face_id and str(state.last_face_id) == str(expected_id)
                            qr_match = state.last_qr_id and str(state.last_qr_id) == str(expected_id)
                            
                            if state.last_face_detected or state.last_qr_detected:
                                if not face_match and not qr_match:
                                    # Both wrong
                                    if state.last_face_id and state.last_qr_id:
                                        detail = "Wrong Face & QR"
                                        color = (0, 0, 255)  # Red
                                    elif state.last_face_id:
                                        detail = "Wrong Face, QR Required"
                                        color = (0, 165, 255)  # Orange
                                    elif state.last_qr_id:
                                        detail = "Face Required, Wrong QR"
                                        color = (0, 165, 255)  # Orange
                                    else:
                                        detail = "Scanning..."
                                        color = (255, 255, 0)  # Yellow
                                elif face_match and not qr_match:
                                    # Face correct, QR wrong or missing
                                    if state.last_qr_id:
                                        detail = f"Face OK ({state.last_face_id}), Wrong QR"
                                        color = (0, 165, 255)  # Orange
                                    else:
                                        detail = f"Face OK ({state.last_face_id}), QR Required"
                                        color = (0, 255, 255)  # Cyan
                                elif not face_match and qr_match:
                                    # QR correct, face wrong or missing
                                    if state.last_face_id:
                                        detail = f"Wrong Face, QR OK ({state.last_qr_id})"
                                        color = (0, 165, 255)  # Orange
                                    else:
                                        detail = f"Face Required, QR OK ({state.last_qr_id})"
                                        color = (0, 255, 255)  # Cyan
                                else:
                                    # Both correct (will verify next)
                                    detail = f"Face & QR Matched ({expected_id})"
                                    color = (0, 255, 0)  # Green
                                
                                cv.putText(disp, detail, (10, 75), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            else:
                                cv.putText(disp, "Scanning...", (10, 75), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                        else:
                            # No person in zone
                            cv.putText(disp, f"Wait: {student['name']} - No person in zone", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 2)
                    else:
                        # Person tracking disabled
                        cv.putText(disp, f"Wait:{student['name']}", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
            
            elif state.verification_state == 'displaying':
                remaining = state.display_duration - (now - state.display_start_time)
                if remaining <= 0:
                    # Countdown finished, move to next student
                    state.current_queue_index += 1
                    state.verification_state = 'waiting'
                    state.verified_student = None
                    state.last_face_id = None
                    state.last_face_box = None
                    state.last_face_detected = False
                    state.last_qr_id = None
                    state.last_qr_rect = None
                    state.last_qr_detected = False
                    # Reset person tracker state for the verified track
                    if person_tracker and person_in_zone:
                        person_tracker.reset_track_state(person_in_zone['track_id'])
                
                # Draw frame during display (show verified student info)
                if state.person_tracking_enabled and person_tracker:
                    disp = person_tracker.draw_tracking(frame)
                else:
                    disp = draw_detections(frame, state.last_face_id, state.last_face_box, 
                                           state.last_qr_id, state.last_qr_rect)
                cv.putText(disp, f"FPS:{fps} | {algo}", (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                cv.putText(disp, f"VERIFIED: {student['name']}", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                cv.putText(disp, f"Next in: {int(remaining)}s", (10, 75), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
            
            else:
                # Fallback
                disp = frame.copy()
        
        _, buffer = cv.imencode('.jpg', disp, [cv.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    camera.stop()

# ==================== ROUTES ====================
@app.route('/')
def index():
    return render_template('index.html', algorithms=available_algorithms)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/start', methods=['POST'])
def start_queue():
    global state
    with state.lock:
        if len(state.queue_list) > 0:
            if camera.start():
                state.queue_started = True
                state.verification_state = 'waiting'
                state.current_queue_index = 0  # Reset to first student
                state.verified_student = None  # Clear any previous verified student
                state.last_face_id = None
                state.last_face_box = None
                state.last_face_detected = False
                state.last_qr_id = None
                state.last_qr_rect = None
                state.last_qr_detected = False
                return jsonify({'success': True})
            return jsonify({'success': False, 'error': 'Camera failed'})
        return jsonify({'success': False, 'error': 'Queue empty'})

@app.route('/api/stop', methods=['POST'])
def stop_queue():
    global state
    with state.lock:
        state.queue_started = False
        state.verification_state = 'waiting'
    camera.stop()
    return jsonify({'success': True})

@app.route('/api/skip', methods=['POST'])
def skip_current():
    global state
    with state.lock:
        if state.current_queue_index < len(state.queue_list):
            state.current_queue_index += 1
            state.verification_state = 'waiting'
            state.verified_student = None
            state.last_face_id = None
            state.last_face_box = None
            state.last_face_detected = False
            state.last_qr_id = None
            state.last_qr_rect = None
            state.last_qr_detected = False
            # Check if we've finished the queue
            if state.current_queue_index >= len(state.queue_list):
                state.queue_started = False
            return jsonify({'success': True})
        return jsonify({'success': False, 'error': 'Queue finished'})

@app.route('/api/status')
def get_status():
    global state, person_tracker
    with state.lock:
        current_student = None
        if state.queue_list and state.current_queue_index < len(state.queue_list):
            current_student = state.queue_list[state.current_queue_index]
        
        remaining = 0
        if state.verification_state == 'displaying':
            remaining = max(0, state.display_duration - (time.time() - state.display_start_time))
        
        verified_img = None
        grad_level = None
        if state.verified_student:
            img_path = find_student_image(state.verified_student['student_id'])
            if img_path:
                verified_img = '/' + img_path.replace('\\', '/')
            grad_level = graduation_level(state.verified_student.get('cgpa', 0) or 0)

        # Get person tracking stats
        tracking_stats = None
        if person_tracker:
            tracking_stats = person_tracker.get_stats()

        return jsonify({
            'queue_started': state.queue_started,
            'current_index': state.current_queue_index,
            'total': len(state.queue_list),
            'verification_state': state.verification_state,
            'remaining': int(remaining),
            'current_student': current_student,
            'verified_student': state.verified_student,
            'verified_img': verified_img,
            'graduation_level': grad_level,
            'face_detected': state.last_face_detected,
            'face_id': state.last_face_id,
            'qr_detected': state.last_qr_detected,
            'qr_id': state.last_qr_id,
            'algorithm': state.current_algorithm,
            'person_tracking_enabled': state.person_tracking_enabled,
            'tracking_stats': tracking_stats
        })

@app.route('/api/settings', methods=['POST'])
def update_settings():
    global state
    data = request.json
    with state.lock:
        if 'algorithm' in data:
            new_algo = data['algorithm']
            if new_algo != state.current_algorithm:
                state.current_algorithm = new_algo
                # Clear detection state when switching algorithm
                state.last_face_id = None
                state.last_face_box = None
                state.last_face_detected = False
                print(f"🔄 Switched to {new_algo}")
                
                # Refresh queue from database to sync attended status
                refreshed_queue = []
                for student in state.queue_list:
                    db_student = db.get_student(student['student_id'])
                    if db_student:
                        refreshed_queue.append(db_student)
                    else:
                        refreshed_queue.append(student)
                state.queue_list = refreshed_queue
            
        if 'tts_enabled' in data:
            state.tts_enabled = data['tts_enabled']
        
        # NEW: Toggle person tracking
        if 'person_tracking_enabled' in data:
            state.person_tracking_enabled = data['person_tracking_enabled']
            print(f"🔄 Person tracking: {'enabled' if state.person_tracking_enabled else 'disabled'}")
    
    return jsonify({'success': True})

@app.route('/api/queue', methods=['GET'])
def get_queue():
    global state
    with state.lock:
        # Refresh attended status from database
        refreshed_queue = []
        for student in state.queue_list:
            db_student = db.get_student(student['student_id'])
            if db_student:
                refreshed_queue.append(db_student)
            else:
                refreshed_queue.append(student)
        state.queue_list = refreshed_queue
        
        return jsonify({
            'queue': state.queue_list,
            'current_index': state.current_queue_index
        })

@app.route('/api/queue/add', methods=['POST'])
def add_to_queue():
    global state
    data = request.json
    student_id = data.get('student_id')
    student = db.get_student(student_id)
    if student:
        with state.lock:
            state.queue_list.append(student)
            if state.email_enabled:
                email_sender = get_email_sender()
                email_sender.send_qr_email_async(student)
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Student not found'})

@app.route('/api/queue/add_all', methods=['POST'])
def add_all_to_queue():
    global state
    with state.lock:
        all_students = db.get_all_students()
        # Only add students who haven't attended
        new_queue = [s for s in all_students if not s.get('attended', False)]
        state.queue_list = new_queue

        if state.email_enabled:
            email_sender = get_email_sender()
            for student in new_queue:
                email_sender.send_qr_email_async(student)
    return jsonify({'success': True})

@app.route('/api/queue/clear', methods=['POST'])
def clear_queue():
    global state
    with state.lock:
        state.queue_list = []
        state.current_queue_index = 0
        state.verification_state = 'waiting'
        state.verified_student = None
        state.last_face_id = None
        state.last_face_box = None
        state.last_face_detected = False
        state.last_qr_id = None
        state.last_qr_rect = None
        state.last_qr_detected = False
    return jsonify({'success': True})

@app.route('/api/queue/remove', methods=['POST'])
def remove_from_queue():
    global state
    data = request.json
    idx = data.get('index')
    with state.lock:
        if 0 <= idx < len(state.queue_list):
            state.queue_list.pop(idx)
            if idx < state.current_queue_index:
                state.current_queue_index -= 1
            return jsonify({'success': True})
    return jsonify({'success': False})

@app.route('/api/students')
def get_students():
    show_attended = request.args.get('show_attended', 'true').lower() == 'true'
    students = db.get_all_students()
    if not show_attended:
        students = [s for s in students if not s.get('attended', False)]
    return jsonify(students)
@app.route('/api/stats')
def get_stats():
    stats = db.get_attendance_stats()
    return jsonify(stats)

@app.route('/api/reset_attendance', methods=['POST'])
def reset_attendance():
    global state
    db.reset_all_attendance()
    with state.lock:
        state.current_queue_index = 0
        state.verification_state = 'waiting'
        state.verified_student = None
    return jsonify({'success': True})

@app.route('/data/Image/<path:filename>')
def serve_image(filename):
    from flask import send_from_directory
    return send_from_directory('data/Image', filename)


@app.route('/api/email/toggle', methods=['POST'])
def toggle_email():
    """Enable/disable email sending"""
    global state
    data = request.json
    with state.lock:
        state.email_enabled = data.get('enabled', False)
    return jsonify({'success': True, 'email_enabled': state.email_enabled})

@app.route('/api/email/status')
def email_status():
    """Get email configuration status"""
    global state
    return jsonify({
        'enabled': state.email_enabled
    })


if __name__ == '__main__':
    app.run(debug=False, threaded=True, host='0.0.0.0', port=5000)