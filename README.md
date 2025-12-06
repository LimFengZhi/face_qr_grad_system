```markdown
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

---

## 🚀 What's New in v2.0

### 🎯 YOLOv8 Person Tracking
- **Zone-based verification**: Detects when person enters verification zone (30%-70% horizontal, 10%-90% vertical)
- **Centroid tracking**: Maintains person ID across frames
- **Single person mode**: `max_tracks=1` for focused scanning
- **Smart detection**: Face/QR scanning only activates when person is in zone

### 📱 LED-Optimized QR Scanner
- **Glare removal**: Morphological top-hat + inpainting for bright LED screens
- **Flicker reduction**: Median filtering for LED refresh artifacts
- **Illumination normalization**: Handles uneven lighting and spotlights
- **Multi-scale pyramid**: 6 scales (1x to 5x) for small and distant QR codes
- **Region detection**: Automatically finds and super-upscales tiny QR codes (3% of frame)
- **4-step scanning strategy**: Direct → LED preprocess → Multi-scale → Otsu threshold

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

## 📸 Screenshots

<p align="center">
  <i>Add screenshots of your application here</i>
</p>

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
<summary>❌ InsightFace installation fails</summary>

```powershell
pip install installation_tools/insightface-0.7.3-cp310-cp310-win_amd64.whl
```
</details>

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

### Train Face Encodings

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
│   ├── 📂 encodings/            # Face encoding files (.pkl)
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
│       ├── person_tracker.py        # 🆕 YOLOv8 person tracking
│       ├── qr_code_scanner.py       # 🆕 LED-optimized QR scanner
│       ├── student_database.py
│       └── text_to_speach.py
│
├── 📂 static/                   # CSS & JavaScript
├── 📂 templates/                # HTML templates
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

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/video_feed` | GET | MJPEG video stream with person tracking |
| `/api/start` | POST | Start queue processing |
| `/api/stop` | POST | Stop queue processing |
| `/api/skip` | POST | Skip current student |
| `/api/status` | GET | Get current status (includes person_in_zone) |
| `/api/queue` | GET | Get queue list |
| `/api/queue/add` | POST | Add student to queue |
| `/api/queue/add_all` | POST | Add all students |
| `/api/queue/clear` | POST | Clear queue |
| `/api/students` | GET | Get all students |
| `/api/stats` | GET | Get attendance stats |
| `/api/reset_attendance` | POST | Reset all attendance |

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
3. Test components individually (test_qr.py)
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
  <sub>Version 2.0 - Now with AI Person Tracking & LED-Optimized QR Scanning</sub>
</p>
