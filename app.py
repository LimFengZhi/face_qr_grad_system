import streamlit as st
import cv2 as cv
import numpy as np
import pandas as pd
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from img_processing_class.utility_class.adaptive_preprocessor import AdaptivePreprocessor
from img_processing_class.utility_class.qr_code_scanner import QRCodeScanner
from img_processing_class.utility_class.box_helper import draw_detections

# Page config
st.set_page_config(
    page_title="Face + QR Verification",
    page_icon="🔐",
    layout="wide"
)

# ==================== SESSION STATE ====================
if 'face_id' not in st.session_state:
    st.session_state.face_id = None
if 'qr_id' not in st.session_state:
    st.session_state.qr_id = None
if 'verified' not in st.session_state:
    st.session_state.verified = False
if 'student_info' not in st.session_state:
    st.session_state.student_info = None
if 'face_detected' not in st.session_state:
    st.session_state.face_detected = False
if 'qr_detected' not in st.session_state:
    st.session_state.qr_detected = False
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'current_algorithm' not in st.session_state:
    st.session_state.current_algorithm = None
if 'camera_initialized' not in st.session_state:
    st.session_state.camera_initialized = False

# ==================== THREAD POOL FOR PARALLEL PROCESSING ====================
executor = ThreadPoolExecutor(max_workers=4)

# ==================== CAMERA MANAGEMENT (SINGLETON) ====================
@st.cache_resource
def get_camera():
    """Initialize camera ONCE and cache it globally"""
    print("🎥 Initializing camera...")
    cap = None
    
    try:
        cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        if cap.isOpened():
            cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv.CAP_PROP_FPS, 30)
            cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
            print("✓ Camera opened with CAP_DSHOW")
            return cap
    except Exception as e:
        print(f"CAP_DSHOW failed: {e}")
        if cap:
            cap.release()
    
    try:
        cap = cv.VideoCapture(0)
        if cap.isOpened():
            cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
            print("✓ Camera opened with default backend")
            return cap
    except Exception as e:
        print(f"Default backend failed: {e}")
        if cap:
            cap.release()
    
    return None

# ==================== 1. PARALLEL MODEL LOADING ====================
def load_hog_dlib():
    """Load HOG + Dlib model"""
    try:
        from img_processing_class.fr_algorithm_class.fr_hog_dlib import FaceRecognitionHogDlib
        model = FaceRecognitionHogDlib(
            file_path="encodings/hb_encoding.pkl",
            confidence=0.6
        )
        print("✓ HOG + Dlib loaded")
        return ("HOG + Dlib", model)
    except Exception as e:
        print(f"✗ HOG + Dlib failed: {e}")
        return ("HOG + Dlib", None)

def load_deepface():
    """Load DeepFace model"""
    try:
        from img_processing_class.fr_algorithm_class.fr_deepface import FaceRecognitionDeepFace
        model = FaceRecognitionDeepFace(
            file_path="encodings/deepface_facenet512.pkl",
            threshold=0.68,
            model_name='Facenet512',
            detector_backend='retinaface'
        )
        print("✓ DeepFace loaded")
        return ("DeepFace", model)
    except Exception as e:
        print(f"✗ DeepFace failed: {e}")
        return ("DeepFace", None)

def load_insightface():
    """Load InsightFace model"""
    try:
        from img_processing_class.fr_algorithm_class.fr_insightface import FaceRecognitionInsightFace
        model = FaceRecognitionInsightFace(
            file_path="encodings/insightface_buffalo.pkl",
            threshold=0.3,
            model_name='buffalo_s',
            ctx_id=-1
        )
        print("✓ InsightFace loaded")
        return ("InsightFace", model)
    except Exception as e:
        print(f"✗ InsightFace failed: {e}")
        return ("InsightFace", None)

def load_mtcnn_facenet():
    """Load MTCNN + FaceNet model"""
    try:
        from img_processing_class.fr_algorithm_class.fr_mtcnn_facenet import FaceRecognitionMTCNNFaceNet
        model = FaceRecognitionMTCNNFaceNet(
            file_path="encodings/mtcnn_facenet.pkl",
            threshold=0.4
        )
        print("✓ MTCNN + FaceNet loaded")
        return ("MTCNN + FaceNet", model)
    except Exception as e:
        print(f"✗ MTCNN + FaceNet failed: {e}")
        return ("MTCNN + FaceNet", None)

@st.cache_resource
def load_all_models_parallel():
    """Load all face recognition models - MTCNN loaded separately due to TensorFlow threading issues"""
    models = {}
    
    futures = [
        executor.submit(load_hog_dlib),
        executor.submit(load_deepface),
        executor.submit(load_insightface),
    ]
    
    for future in as_completed(futures):
        name, model = future.result()
        models[name] = model
    
    # Load MTCNN + FaceNet SEQUENTIALLY
    name, model = load_mtcnn_facenet()
    models[name] = model
    
    return models

@st.cache_resource
def load_preprocessor():
    return AdaptivePreprocessor()

@st.cache_resource
def load_qr_scanner():
    return QRCodeScanner()

@st.cache_data
def load_student_data(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def build_student_lookup(_df):
    """Build a dictionary for O(1) student lookup"""
    if _df is None:
        return {}
    
    lookup = {}
    id_columns = ['student_id', 'id', 'ID', 'StudentID']
    
    for col in id_columns:
        if col in _df.columns:
            for idx, row in _df.iterrows():
                lookup[str(row[col])] = row.to_dict()
            break
    
    return lookup

def get_student_info_fast(student_id, lookup):
    return lookup.get(str(student_id))

def detect_face_pipeline(frame, model, preprocessor, algo_name):
    """Detect and recognize face"""
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
        
        else:
            rgb_frame = cv.cvtColor(processed, cv.COLOR_BGR2RGB)
            face_loc = model.detect_face(rgb_frame)
            
            if face_loc is None or len(face_loc) == 0:
                return None, None, False
            
            if isinstance(face_loc, list) and len(face_loc) > 0:
                face_loc = max(face_loc, key=lambda f: (f[2] - f[0]) * (f[1] - f[3]))
            
            if algo_name == "HOG + Dlib":
                top, right, bottom, left = face_loc
                box = (left, top, right, bottom)
            else:
                if hasattr(face_loc, 'get'):
                    fa = face_loc.get('facial_area', face_loc.get('box', {}))
                    if 'x' in fa:
                        box = (fa['x'], fa['y'], fa['x'] + fa['w'], fa['y'] + fa['h'])
                    else:
                        top, right, bottom, left = face_loc
                        box = (left, top, right, bottom)
                else:
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

def detect_qr_pipeline(frame, scanner):
    """Detect QR code"""
    try:
        qr_results = scanner.scan(frame)
        if qr_results:
            qr_id = qr_results[0]['data']
            rect = qr_results[0]['rect']
            qr_rect = (rect.left, rect.top, rect.width, rect.height)
            return qr_id, qr_rect, True
    except:
        pass
    return None, None, False

def detect_parallel(frame, face_model, preprocessor, algo_name, qr_scanner):
    """Run face and QR detection in parallel"""
    face_future = executor.submit(detect_face_pipeline, frame.copy(), face_model, preprocessor, algo_name)
    qr_future = executor.submit(detect_qr_pipeline, frame.copy(), qr_scanner)
    
    face_result = face_future.result()
    qr_result = qr_future.result()
    
    return face_result, qr_result

def find_student_image(student_id):
    """Find student image"""
    image_paths = [
        f"Image/{student_id}.jpg",
        f"Image/{student_id}.png", 
        f"Image/{student_id}.jpeg",
        f"Image/{student_id}.JPG",
        f"Image/{student_id}.PNG",
    ]
    for img_path in image_paths:
        if os.path.exists(img_path):
            return img_path
    return None

def load_student_image_async(student_id):
    """Load student image asynchronously"""
    future = executor.submit(find_student_image, student_id)
    return future

# ==================== LOAD ALL RESOURCES AT STARTUP ====================
if not st.session_state.models_loaded:
    with st.spinner("🚀 Loading all models in parallel..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Starting parallel loading...")
        progress_bar.progress(10)
        
        preprocessor_future = executor.submit(load_preprocessor.__wrapped__)
        qr_future = executor.submit(load_qr_scanner.__wrapped__)
        
        status_text.text("Loading all face recognition models in parallel...")
        progress_bar.progress(30)
        
        all_models = load_all_models_parallel()
        
        preprocessor = preprocessor_future.result()
        qr_scanner = qr_future.result()
        
        progress_bar.progress(90)
        status_text.text("Initializing camera...")
        
        # Initialize camera during loading
        cap = get_camera()
        
        progress_bar.progress(100)
        status_text.text("✅ All models loaded!")
        time.sleep(0.3)
        progress_bar.empty()
        status_text.empty()
        
        st.session_state.models_loaded = True
else:
    preprocessor = load_preprocessor()
    qr_scanner = load_qr_scanner()
    all_models = load_all_models_parallel()

# Get cached camera (won't reinitialize)
cap = get_camera()

# Get list of available models
available_algorithms = [name for name, model in all_models.items() if model is not None]

# ==================== SIDEBAR ====================
st.sidebar.header("🔄 Switch Algorithm")

algorithm = st.sidebar.selectbox(
    "Select Face Recognition",
    available_algorithms,
    index=0
)
st.session_state.current_algorithm = algorithm

csv_path = st.sidebar.text_input("CSV File Path", value="student_list.csv")
student_df = load_student_data(csv_path)
student_lookup = build_student_lookup(student_df)

st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Performance")
process_every_n = st.sidebar.slider("Process every N frames", 1, 10, 2)

# Get selected model
face_model = all_models.get(algorithm)

# ==================== 2-COLUMN LAYOUT ====================
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader(f"📹 Face + QR Scanner ({algorithm})")
    FRAME_WINDOW = st.empty()
    
    st.markdown("---")
    st.subheader("📊 Detection Status")
    status_col1, status_col2 = st.columns(2)
    face_status = status_col1.empty()
    qr_status = status_col2.empty()
    match_status = st.empty()

with col_right:
    st.subheader("👤 Student Info")
    student_image_placeholder = st.empty()
    student_info_placeholder = st.empty()

# ==================== MAIN CAMERA LOOP ====================
if cap is not None and cap.isOpened():
    # Flush old frames
    for _ in range(3):
        cap.grab()
    
    frame_count = 0
    last_face_id = None
    last_face_box = None
    last_face_detected = False
    last_qr_id = None
    last_qr_rect = None
    last_qr_detected = False
    error_count = 0
    max_errors = 10
    
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0
    
    image_future = None
    last_verified_id = None
    
    while True:
        try:
            ret, frame = cap.read()
            
            if not ret or frame is None:
                error_count += 1
                if error_count > max_errors:
                    FRAME_WINDOW.error("❌ Camera disconnected. Please refresh the page.")
                    # Clear cache to allow camera reinit on next run
                    get_camera.clear()
                    break
                time.sleep(0.01)
                continue
            
            error_count = 0
            frame_count += 1
            fps_frame_count += 1
            
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                current_fps = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_start_time = time.time()
            
            do_detection = (frame_count % process_every_n == 0)
            
            if do_detection:
                last_face_id, last_face_box, last_face_detected = detect_face_pipeline(
                        frame, face_model, preprocessor, algorithm
                    )
                last_qr_id, last_qr_rect, last_qr_detected = detect_qr_pipeline(frame, qr_scanner)
            
            verified = False
            student_info = None
            if last_face_id and last_qr_id and str(last_face_id) == str(last_qr_id):
                verified = True
                student_info = get_student_info_fast(last_face_id, student_lookup)
                
                if last_face_id != last_verified_id:
                    image_future = load_student_image_async(last_face_id)
                    last_verified_id = last_face_id
            
            display_frame = draw_detections(frame, last_face_id, last_face_box, last_qr_id, last_qr_rect)
            cv.putText(display_frame, f"FPS: {current_fps:.1f}", (10, 30), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            display_frame = cv.cvtColor(display_frame, cv.COLOR_BGR2RGB)
            
            FRAME_WINDOW.image(display_frame, channels="RGB", use_container_width=True)
            
            if frame_count % 3 == 0:
                if last_face_detected:
                    if last_face_id:
                        face_status.success(f"✅ Face: **{last_face_id}**")
                    else:
                        face_status.warning("⚠️ Face: **Unknown**")
                else:
                    face_status.error("❌ No Face")
                
                if last_qr_detected:
                    qr_status.success(f"✅ QR: **{last_qr_id}**")
                else:
                    qr_status.error("❌ No QR")
                
                if verified:
                    match_status.success("### ✅ VERIFIED")
                elif last_face_id and last_qr_id:
                    match_status.error("### ❌ MISMATCH")
                else:
                    match_status.info("### ⏳ Waiting...")
                
                if verified and last_face_id:
                    if image_future and image_future.done():
                        img_path = image_future.result()
                        if img_path:
                            student_image_placeholder.image(img_path, use_container_width=True)
                        else:
                            student_image_placeholder.info("📷 No image")
                    
                    if student_info:
                        with student_info_placeholder.container():
                            show_columns = ['student_id', 'name', 'course', 'cgpa']
                            for k, v in student_info.items():
                                if k in show_columns:
                                    st.write(f"**{k}:** {v}")
                else:
                    student_image_placeholder.info("📷 Waiting...")
                    student_info_placeholder.info("📋 Scan face + QR")
                
        except Exception as e:
            error_count += 1
            if error_count > max_errors:
                FRAME_WINDOW.error(f"❌ Error: {str(e)}")
                break
            continue

else:
    FRAME_WINDOW.error("❌ Cannot open camera")
    st.error("""
    **Troubleshooting:**
    1. Close other apps using camera
    2. Check camera connection
    3. Refresh page (Ctrl+R)
    """)
    # Clear camera cache on error
    get_camera.clear()
    face_status.error("❌ Camera Error")
    qr_status.error("❌ Camera Error")
    match_status.error("### ❌ Camera Error")