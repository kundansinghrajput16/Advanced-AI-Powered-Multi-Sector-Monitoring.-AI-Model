# 🚨 Sentrix – AI-Based Smart Surveillance System

## 📌 Overview

**Sentrix** is an AI-powered real-time surveillance system designed to enhance safety using computer vision and machine learning. It detects **people, weapons, crowd density, and garbage objects** through live video streams and triggers alerts with snapshot storage.

The system integrates multiple ML models to monitor environments and automatically capture critical events for further analysis.

---

## 🧠 Machine Learning Component

The core of Sentrix is built using **YOLOv8 (You Only Look Once)** for real-time object detection.

### 🔍 Models Used

1. **Pre-trained YOLOv8 Nano Model (`yolov8n.pt`)**

   * Used for:

     * Person detection
     * Crowd counting
   * Lightweight and optimized for real-time inference

2. **Custom Trained YOLOv8 Model**

   * Path: `runs/detect/weapon_detector/weights/best.pt`
   * Used for:

     * Weapon detection (knife, handgun, rifle)
     * Garbage classification (multi-class detection)

---

## ⚙️ ML Pipeline

### 1. Input Stream

* Live video feed via IP webcam (`cv2.VideoCapture`)
* Frames processed in real-time

---

### 2. Object Detection

* Each frame is passed through YOLO models
* Bounding boxes, class labels, and confidence scores are extracted

Example:

* Person detection with confidence > 0.6
* Weapon detection for specific classes: *knife, handgun, rifle*



---

### 3. Multi-Condition Logic

The system intelligently combines detections:

* **Person only → Normal monitoring**
* **Weapon only → Suspicious object alert**
* **Person + Weapon → High-risk alert**

This fusion enables contextual decision-making instead of isolated detection.

---

### 4. Crowd Detection Logic

* Counts number of detected persons per frame
* Triggers alert when:

  ```
  person_count > threshold (e.g., 3)
  ```
* Includes cooldown mechanism to prevent alert spam



---

### 5. Garbage Detection (Custom ML Use Case)

* Uses trained YOLO model for multi-class garbage detection
* Tracks **newly appearing objects** using set comparison:

  ```
  new_objects = detected_objects - previous_objects
  ```
* Captures image only when new garbage is detected



---

### 6. Event Trigger & Storage

* When an event is detected:

  * Frame is saved as an image
  * Stored in Google Drive directory
* Includes cooldown timing to avoid redundant captures

---

## 🧪 Key ML Features

* ✅ Real-time object detection (low latency)
* ✅ Multi-model inference (person + weapon)
* ✅ Custom-trained dataset support
* ✅ Event-based image capture
* ✅ Context-aware alert system
* ✅ Edge-device friendly (YOLOv8n)

---

## 🏗️ Tech Stack

* **Python**
* **OpenCV** – Video processing
* **Ultralytics YOLOv8** – Object detection
* **NumPy (implicit)** – Data handling
* **Google Drive API (storage path usage)**

---

## 📊 Use Cases

* 🔐 Smart surveillance systems
* 🏙️ Smart city monitoring
* 🚔 Crime detection (weapon + person)
* 🧍 Crowd management
* ♻️ Waste detection & monitoring

---

## 🚀 Future Improvements

* Integration with alert systems (SMS/Telegram bot)
* Cloud dashboard for analytics
* Model optimization (TensorRT / ONNX)
* Edge deployment on Jetson Nano / Raspberry Pi
* Dataset expansion for higher accuracy

---

## 👨‍💻 Author

**Kundan Raj Singh**
