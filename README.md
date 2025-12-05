# 🎓 Face & QR Graduate Ceremony Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10.8-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Flask-3.0.3-green?style=for-the-badge&logo=flask" alt="Flask">
  <img src="https://img.shields.io/badge/OpenCV-4.12-red?style=for-the-badge&logo=opencv" alt="OpenCV">
  <img src="https://img.shields.io/badge/TensorFlow-2.13-orange?style=for-the-badge&logo=tensorflow" alt="TensorFlow">
</p>

<p align="center">
  A real-time face recognition and QR code scanning system for graduate ceremonies with Text-to-Speech (TTS) announcements.
</p>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **Multi-Algorithm Face Recognition** | HOG+Dlib, DeepFace, InsightFace, MTCNN+FaceNet |
| 📱 **QR Code Scanning** | Fast QR code detection with zoom enhancement |
| 🔊 **TTS Announcements** | "Congratulations [Name], graduated with [Distinction/Merit/Pass]" |
| 📧 **Email Notifications** | Auto-send QR codes to students |
| 🌐 **Web Interface** | Modern Flask-based UI with real-time video |
| 📊 **Attendance Tracking** | SQLite database with statistics |
| 🔄 **Queue Management** | Add, remove, skip students in queue |

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

### Step 3: Troubleshooting

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

---

## 💻 Usage

### Run the Application

```powershell
python app.py
```

Then open your browser: **http://localhost:5000**

### Train Face Encodings

1. Add student images to `data/Image/` folder (named as `student_id.jpg`)
2. Update `data/student_list.csv` with student information
3. Open `train.ipynb` in Jupyter Notebook
4. Run all cells to generate encodings

---

## 📁 Project Structure

```
face_qr_grad_system/
│
├── 📄 app.py                    # Main Flask application
├── 📄 requirements.txt          # Python dependencies
├── 📓 train.ipynb               # Encoding training notebook
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
│       ├── qr_code_scanner.py
│       ├── student_database.py
│       └── text_to_speach.py
│
├── 📂 static/                   # CSS & JavaScript
├── 📂 templates/                # HTML templates
└── 📂 installation_tools/       # Pre-built wheels
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

1. Open `img_processing_class/utility_class/email_sender.py`
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
| `/video_feed` | GET | MJPEG video stream |
| `/api/start` | POST | Start queue processing |
| `/api/stop` | POST | Stop queue processing |
| `/api/skip` | POST | Skip current student |
| `/api/status` | GET | Get current status |
| `/api/queue` | GET | Get queue list |
| `/api/queue/add` | POST | Add student to queue |
| `/api/queue/add_all` | POST | Add all students |
| `/api/queue/clear` | POST | Clear queue |
| `/api/students` | GET | Get all students |
| `/api/stats` | GET | Get attendance stats |
| `/api/reset_attendance` | POST | Reset all attendance |

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Lim Feng Zhi**

- GitHub: [@LimFengZhi](https://github.com/LimFengZhi)

---

<p align="center">
  Made with ❤️ for Graduate Ceremonies
</p>