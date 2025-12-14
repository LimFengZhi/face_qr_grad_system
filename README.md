# 🎓 Face & QR Graduate Ceremony Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10.8-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Flask-3.0.3-green?style=for-the-badge&logo=flask" alt="Flask">
  <img src="https://img.shields.io/badge/OpenCV-4.12-red?style=for-the-badge&logo=opencv" alt="OpenCV">
  <img src="https://img.shields.io/badge/TensorFlow-2.13-orange?style=for-the-badge&logo=tensorflow" alt="TensorFlow">
  <img src="https://img.shields.io/badge/YOLOv8-ultralytics-purple?style=for-the-badge" alt="YOLOv8">
</p>

<p align="center">
  A real-time intelligent face recognition and QR code scanning system for graduate ceremonies with AI-powered person tracking and Text-to-Speech announcements.
</p>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **Multi-Algorithm Face Recognition** | HOG+Dlib, DeepFace, InsightFace, MTCNN+FaceNet |
| 👤 **YOLOv8 Person Tracking** | Zone-based detection with centroid tracking (max 1 person) |
| 📱 **Advanced QR Code Scanning** | LED-optimized with glare removal, multi-scale detection for small/distant codes |
| ⚡ **Performance Optimized** | Parallel face+QR detection, frame skipping, 20-25 FPS |
| 🔊 **TTS Announcements** | "Congratulations [Name], graduated with [Distinction/Merit/Pass]" |
| 📧 **Email Notifications** | Auto-send QR codes to students |
| 🌐 **Web Interface** | Modern Flask-based UI with real-time video streaming |
| 📊 **Attendance Tracking** | SQLite database with comprehensive statistics |
| 🔄 **Queue Management** | Add, remove, skip students in queue |
| 🎨 **Color-Coded Feedback** | Green (match), Red (wrong), Orange (unknown) visual indicators |
| 📝 **Student Registration** | 🆕 Web-based registration with face capture & validation |
| 🗑️ **Student Management** | 🆕 Delete students (database, encodings, images, QR codes) |

---

## 🆕 New Features (v2.1)

### 📝 Student Registration System
- **Web-based registration page** at `/register`
- **3-step registration process**:
  1. Fill student details (ID, Name, Email, Faculty, Course, CGPA)
  2. Capture face via webcam
  3. Complete registration
- **Multi-algorithm registration**: Face encoded in ALL available algorithms simultaneously
- **Duplicate detection**: Prevents same face from registering twice
- **Auto QR generation**: QR code automatically created on registration
- **Real-time validation**: Student ID uniqueness, email format, required fields

### 🗑️ Student Deletion
- **Complete removal**: Deletes from database, queue, ALL algorithm encodings, images, and QR codes
- **Dedicated UI section**: Safe deletion from Queue Management tab
- **Confirmation dialog**: Prevents accidental deletions

---

### 🎯 YOLOv8 Person Tracking
- **Zone-based verification**: Detects when person enters verification zone (30%-70% horizontal, 10%-90% vertical)
- **Centroid tracking**: Maintains person ID across frames
- **Single person mode**: `max_tracks=1` for focused scanning
- **Smart detection**: Face/QR scanning only activates when person is in zone

### ⚡ Performance Improvements
- **Parallel detection**: ThreadPoolExecutor for simultaneous face + QR scanning
- **Frame skipping**: Process every 5th frame (20-25 FPS from 4 FPS)
- **Optimized preprocessing**: Skip heavy processing for HOG+Dlib
- **Smart caching**: Avoid redundant computations

### 🎨 Enhanced UI Feedback
- **Color-coded boxes**: Green (correct match), Red (wrong person), Orange (unknown)
- **Detailed status messages**:
  - ✅ "Face and QR match! Processing..."
  - ⚠️ "Face matches, waiting for QR..."
  - ❌ "QR doesn't match current student"
  - 🔍 "Detecting..."

---

## 🔧 Prerequisites

| Requirement | Version | Download |
|-------------|---------|----------|
| Python | 3.10.8 | [python.org](https://www.python.org/downloads/) |
| C++ Build Tools | Latest | [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) |
| CMake | 3.x | [cmake.org](https://cmake.org/download/) |

---

## 🚀 Installation

### Step 1: Install Build Tools

1. Download [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Select **"Desktop development with C++"** during installation
3. Install [CMake](https://cmake.org/download/) and add to PATH

### Step 2: Clone & Setup

```powershell
# Clone repository
git clone https://github.com/LimFengZhi/face_qr_grad_system.git
cd face_qr_grad_system

# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Download YOLOv8 Model

The system will automatically download yolov8n.pt on first run, or download manually:

```powershell
# Using Python
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### Step 4: Troubleshooting
<details>
<summary>❌ Dlib installation fails</summary>

```powershell
# Ensure CMake is installed and in PATH
cmake --version

# Try installing dlib separately
pip install dlib
```
</details>

<details>
<summary>❌ Ultralytics/YOLOv8 issues</summary>

```powershell
# Reinstall ultralytics
pip uninstall ultralytics -y
pip install ultralytics --no-cache-dir
```
</details>

---

## 💻 Usage

### Run the Application

```powershell
python app.py
```

Then open your browser: **http://localhost:5000**

### Web Interface Pages

| Page | URL | Description |
|------|-----|-------------|
| 🎓 Scanner | `/` | Main verification scanner |
| 📋 Queue Management | `/` (tab) | Manage queue, statistics, delete students |
| 📝 Registration | `/register` | Register new students |

### Register a New Student

1. Click **"📝 Register"** tab in navigation
2. Fill in student information:
   - Student ID (unique, min 5 characters)
   - Full Name
   - Email
   - Faculty (dropdown)
   - Course
   - CGPA (optional)
3. Position face in camera and click **"📸 Capture Face"**
4. Click **"✅ Complete Registration"**
5. System will:
   - Register face in ALL available algorithms
   - Save student image
   - Generate QR code
   - Add to database

### Delete a Student

1. Go to **"📋 Queue Management"** tab
2. Scroll to **"🗑️ Delete Student"** section
3. Select student from dropdown
4. Click **"🗑️ Delete Permanently"**
5. Confirm deletion
6. System removes:
   - Database record
   - Face encodings (all algorithms)
   - Student image
   - QR code

### Train Face Encodings (Batch)

1. Add student images to Image folder (named as `student_id.jpg`)
2. Update student_list.csv with student information:
   ```csv
   Student_ID,Name,Email,CGPA
   2024001,John Doe,john@example.com,3.85
   2024002,Jane Smith,jane@example.com,3.45
   ```
3. Open train.ipynb in Jupyter Notebook
4. Run all cells to generate encodings

---

## 📁 Project Structure

```
face_qr_grad_system/
│
├── 📄 app.py                    # Main Flask application with person tracking
├── 📄 test_qr.py                # QR scanner testing utility
├── 📄 requirements.txt          # Python dependencies
├── 📓 train.ipynb               # Encoding training notebook
├── 📄 yolov8n.pt                # YOLOv8 nano model
│
├── 📂 data/
│   ├── 📄 student_list.csv      # Student information
│   ├── 📄 students.db           # SQLite database
│   ├── 📂 encodings/            # Face encoding files (.pkl)
│   │   ├── 📂 preprocessed/     # Preprocessed encodings
│   │   └── 📂 raw/              # Raw encodings
│   ├── 📂 Image/                # Student photos
│   ├── 📂 qr_codes/             # Generated QR codes
│   └── 📂 test_set/             # Test images by person
│
├── 📂 img_processing_class/
│   ├── 📂 fr_algorithm_class/   # Face recognition algorithms
│   │   ├── fr_deepface.py
│   │   ├── fr_hog_dlib.py
│   │   ├── fr_insightface.py
│   │   └── fr_mtcnn_facenet.py
│   └── 📂 utility_class/        # Helper utilities
│       ├── adaptive_preprocessor.py
│       ├── box_helper.py
│       ├── camera.py
│       ├── email_sender.py
│       ├── person_tracker.py        # YOLOv8 person tracking
│       ├── qr_code_scanner.py       # LED-optimized QR scanner
│       ├── student_database.py
│       └── text_to_speach.py
│
├── 📂 static/                   # CSS & JavaScript
│   ├── style.css
│   └── script.js
│
├── 📂 templates/                # HTML templates
│   ├── index.html               # Main scanner page
│   └── register.html            # 🆕 Registration page
```

---

## 🧠 Face Recognition Algorithms

### Comparison Table

| Algorithm | Detection | Embedding | Speed | Accuracy | GPU Required |
|-----------|-----------|-----------|:-----:|:--------:|:------------:|
| **HOG + Dlib** | HOG | Dlib ResNet | ⚡⚡⚡ | ⭐⭐⭐ | ❌ |
| **DeepFace** | RetinaFace | Facenet512 | ⚡ | ⭐⭐⭐⭐⭐ | ✅ Recommended |
| **InsightFace** | SCRFD | Buffalo | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | ✅ Optional |
| **MTCNN + FaceNet** | MTCNN | FaceNet | ⚡⚡ | ⭐⭐⭐⭐ | ✅ Optional |

### Algorithm Details

<details>
<summary>🔷 HOG + Dlib</summary>

- **Detection**: Histogram of Oriented Gradients (HOG)
- **Embedding**: Dlib's ResNet (128-dimensional)
- **Pros**: Fast, works on CPU, lightweight
- **Cons**: Less accurate in challenging conditions
- **Best for**: Real-time on low-end hardware
</details>

<details>
<summary>🔷 DeepFace (Facenet512 + RetinaFace)</summary>

- **Detection**: RetinaFace (highly accurate)
- **Embedding**: Facenet512 (512-dimensional)
- **Pros**: Highest accuracy
- **Cons**: Slower processing, needs GPU
- **Best for**: When accuracy is critical
</details>

<details>
<summary>🔷 InsightFace (Buffalo)</summary>

- **Detection**: SCRFD (built-in)
- **Embedding**: ArcFace-based
- **Pros**: Fast + accurate, all-in-one
- **Cons**: Complex setup
- **Best for**: Production with GPU
</details>

<details>
<summary>🔷 MTCNN + FaceNet</summary>

- **Detection**: Multi-task Cascaded CNN
- **Embedding**: FaceNet (128-dimensional)
- **Pros**: Robust to angles/lighting
- **Cons**: Moderate speed
- **Best for**: Variable conditions
</details>

### Selection Guide

```
┌─────────────────────────────────────────────────────────────┐
│                  Which algorithm to use?                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Low-end PC / No GPU? ──────────► HOG + Dlib                │
│                                                              │
│  Need highest accuracy? ────────► DeepFace                  │
│                                                              │
│  Have GPU + need speed? ────────► InsightFace               │
│                                                              │
│  Variable lighting/angles? ─────► MTCNN + FaceNet           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Person Tracking System

### YOLOv8 Configuration

```python
# Verification zone (normalized coordinates)
VERIFICATION_ZONE = {
    'x_min': 0.3,   # 30% from left
    'y_min': 0.1,   # 10% from top
    'x_max': 0.7,   # 70% from left (40% width)
    'y_max': 0.9    # 90% from top (80% height)
}

# Tracking settings
MAX_TRACKS = 1              # Only track one person at a time
CONFIDENCE_THRESHOLD = 0.5  # YOLOv8 detection confidence
```

### How It Works

1. **Person Detection**: YOLOv8 detects people in frame
2. **Zone Check**: System checks if person is in verification zone
3. **Activation**: Face + QR scanning only starts when person enters zone
4. **Color Feedback**: Box color indicates status (green/red/orange)
5. **Smart Processing**: Parallel detection for speed

---

## 📱 QR Code Scanner Features

### LED Screen Optimization

The QR scanner is specifically designed for real-world LED displays:

| Problem | Solution |
|---------|----------|
| 💡 **LED Glare** | Morphological top-hat + inpainting |
| ⚡ **Screen Flicker** | Median filtering (reduces refresh artifacts) |
| 🌓 **Uneven Lighting** | Gaussian illumination normalization |
| 🔍 **Small QR Codes** | Multi-scale pyramid (1x to 5x upscaling) |
| 📏 **Distant Codes** | Region detection + 6x super-upscaling |
| 🎨 **High Contrast** | CLAHE with LED-tuned parameters |

### Scanning Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                  QR Scanning Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 1: Direct Scan (fastest)                              │
│     ↓ Failed                                                │
│  Step 2: LED Preprocessing (glare removal + normalize)       │
│     ↓ Failed                                                │
│  Step 3: Multi-Scale Pyramid (1x, 1.5x, 2x, 3x, 4x, 5x)    │
│     ↓ Failed                                                │
│  Step 4: Region Detection + Super-Upscaling (6x)            │
│     ↓ Failed                                                │
│  Step 5: Aggressive Otsu (last resort)                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📝 Registration System

### Registration Flow

```
┌─────────────────────────────────────────────────────────────┐
│                  Registration Process                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 1: Fill Details                                       │
│     • Student ID (validated for uniqueness)                 │
│     • Name, Email, Faculty, Course, CGPA                    │
│     ↓                                                       │
│  Step 2: Capture Face                                       │
│     • Position face in oval guide                           │
│     • Click "Capture Face"                                  │
│     • System checks if face already registered              │
│     ↓                                                       │
│  Step 3: Complete Registration                              │
│     • Face encoded in ALL algorithms                        │
│     • Image saved to data/Image/                            │
│     • QR code generated in data/qr_codes/                   │
│     • Student added to database                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Validation Rules

| Field | Validation |
|-------|------------|
| Student ID | Required, min 5 chars, must be unique |
| Name | Required, min 2 chars, letters & spaces only |
| Email | Valid email format |
| Faculty | Required (dropdown selection) |
| Course | Required |
| CGPA | Optional, 0.00 - 4.00 |

---

## 🎓 Graduation Level

| CGPA | Level |
|------|-------|
| ≥ 3.67 | 🏆 Distinction |
| ≥ 2.67 | 🥈 Merit |
| ≥ 2.00 | 🥉 Pass |
| < 2.00 | ❌ Fail |

---

## 📧 Email Configuration

To enable email notifications:

1. Open email_sender.py
2. Update credentials:
```python
SENDER_EMAIL = "your_email@gmail.com"
SENDER_PASSWORD = "your_app_password"  # Use Gmail App Password
```

3. For Gmail, generate App Password:
   - Enable 2FA on your Google account
   - Go to [App Passwords](https://myaccount.google.com/apppasswords)
   - Generate password for "Mail"

---

## 🛠️ API Endpoints

### Scanner & Queue

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/video_feed` | GET | MJPEG video stream with person tracking |
| `/api/start` | POST | Start queue processing |
| `/api/stop` | POST | Stop queue processing |
| `/api/skip` | POST | Skip current student |
| `/api/status` | GET | Get current status |
| `/api/queue` | GET | Get queue list |
| `/api/queue/add` | POST | Add student to queue |
| `/api/queue/add_all` | POST | Add all students |
| `/api/queue/clear` | POST | Clear queue |
| `/api/queue/remove` | POST | Remove from queue |
| `/api/students` | GET | Get all students |
| `/api/stats` | GET | Get attendance stats |
| `/api/reset_attendance` | POST | Reset all attendance |

### Registration (🆕)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/register` | GET | Registration page |
| `/api/register/check_id` | POST | Check if student ID exists |
| `/api/register/validate` | POST | Validate form data |
| `/api/register/capture_face` | POST | Capture & verify face |
| `/api/register/submit` | POST | Complete registration |
| `/api/register/video_feed` | GET | Camera feed for registration |

### Student Management (🆕)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/student/delete` | POST | Delete student entirely |

### Email

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/email/toggle` | POST | Enable/disable email |
| `/api/email/status` | GET | Get email status |

---

## ⚙️ Configuration

### Performance Tuning

Edit app.py to adjust performance:

```python
# Frame processing
SKIP_FRAMES = 5  # Process every 5th frame (default)

# Person tracking
tracker = PersonTracker(
    model_path='yolov8n.pt',
    max_tracks=1,           # Number of people to track
    verification_zone=(0.3, 0.1, 0.7, 0.9)  # Zone coordinates
)

# QR Scanner
qr_scanner = QRCodeScanner()  # Auto-optimized for LED screens
```

### Verification Zone Adjustment

Modify the verification zone in person_tracker.py:

```python
self.verification_zone = (
    0.3,  # x_min (30% from left)
    0.1,  # y_min (10% from top)
    0.7,  # x_max (70% from left)
    0.9   # y_max (90% from top)
)
```

---

## 🧪 Testing

### Test QR Scanner

```powershell
python test_qr.py
```

Features:
- Real-time webcam QR detection
- FPS counter
- Detection method display
- 2-second result caching

### Test Face Recognition

Use the test images in test_set:
```
test_set/
├── ch/    # Chen Hwee's test images
├── fz/    # Feng Zhi's test images
├── hy/    # Hao Yi's test images
└── zq/    # Zhi Qiang's test images
```

---

## 📊 Performance Benchmarks

| Component | Without Optimization | With Optimization | Improvement |
|-----------|---------------------|-------------------|-------------|
| Frame Rate | 4 FPS | 20-25 FPS | **500%** ⬆️ |
| Face Detection | Every frame | Every 5th frame | **80% CPU** ⬇️ |
| QR Detection | Sequential | Parallel | **2x faster** ⚡ |
| Person Tracking | N/A | YOLOv8 real-time | **Zone-based** ✅ |
| Registration | Single algo | All algos parallel | **Future-proof** 🔒 |

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **YOLOv8** by Ultralytics for person detection
- **DeepFace** for face recognition framework
- **InsightFace** for high-accuracy face embeddings
- **Dlib** for lightweight face detection
- **Pyzbar** for QR code decoding

---

## 👨‍💻 Author

**Lim Feng Zhi**

- GitHub: [@LimFengZhi](https://github.com/LimFengZhi)

---

## 📞 Support

If you encounter any issues:

1. Check the Troubleshooting section
2. Review API Endpoints documentation
3. Test components individually
4. Open an issue on GitHub with:
   - Error message
   - Python version
   - OS details
   - Steps to reproduce

---

<p align="center">
  Made with ❤️ for Graduate Ceremonies
</p>

<p align="center">
  <sub>Version 2.1 - Now with Student Registration & Management</sub>
</p>
