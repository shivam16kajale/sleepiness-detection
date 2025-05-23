# 💤 Real-Time Drowsiness and Yawn Detection Using OpenCV & MediaPipe

This project is a real-time computer vision system that monitors a user's eyes and mouth via webcam to detect signs of **drowsiness** and **yawning**. If prolonged eye closure is detected, the system plays an audible alert to wake the user.

---

## 📸 Features

- Detects closed eyes using **Eye Aspect Ratio (EAR)**
- Detects yawns using **Mouth Aspect Ratio (MAR)**
- Plays a sound alert if the user appears drowsy
- Displays real-time EAR, MAR, and yawn count
- Works with any standard webcam

---

## 🛠️ Technologies Used

- **Python**
- **OpenCV** – For video capture and image processing
- **MediaPipe** – For facial landmark detection
- **NumPy** – For numeric computations
- **winsound** – For alert sound playback (Windows only)

---

## 🧠 How It Works

### Face Mesh Landmarks
We use MediaPipe's Face Mesh to extract facial landmarks. Specific landmark indices are used to calculate:

- **EAR (Eye Aspect Ratio)** — measures eye openness
- **MAR (Mouth Aspect Ratio)** — measures mouth opening

### Logic:
- If EAR falls below a threshold for several consecutive frames → **Drowsiness Alert**
- If MAR rises above a threshold → **Yawn Detected**

---

## 🔧 Setup Instructions

1. **Install dependencies**
   ```bash
   pip install opencv-python mediapipe numpy
