# 🛡️ Project Sentinel: Deepfake Forensics
### Automated Deepfake Detection & Forensic Analysis System
*(Capstone Project Submission)*

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/YOUR_USERNAME_HERE/Sentinel-Deepfake-Detector)
[![Python](https://img.shields.io/badge/Python-3.10-yellow)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview
**Project Sentinel** is a dual-stream deepfake detection system designed to identify manipulated media in an era of generative AI threats. Unlike traditional detectors that look at single frames, Sentinel analyzes both **spatial artifacts** (texture/pixel inconsistencies) and **temporal inconsistencies** (movement glitches over time) to provide a robust forensic verdict.

The system features an **Explainable AI (XAI)** layer using Grad-CAM, which highlights *where* the manipulation was detected, making it suitable for forensic auditing rather than just black-box prediction.


---

## System Architecture

The project utilizes a hybrid deep learning architecture:

1.  **Spatial Stream (The "Eye"):** * **Model:** EfficientNet-B0 (Pretrained on ImageNet).
    * **Function:** Detects pixel-level artifacts, blending glitches, and unnatural textures in individual frames.
    * **Focus:** Skin texture analysis and boundary detection.

2.  **Temporal Stream (The "Brain"):**
    * **Model:** CNN Feature Extractor + LSTM (Long Short-Term Memory).
    * **Function:** Analyzes a sequence of 10 frames to detect "temporal jitter"—unnatural mouth movements or flickering that occurs in Deepfakes but not in real video.

3.  **Explainability Layer:**
    * **Technique:** Grad-CAM (Gradient-weighted Class Activation Mapping).
    * **Output:** Generates a heat map overlay showing the specific facial regions contributing to the "Fake" verdict.

---

## Key Features
* **Dual-Input Support:** Accepts both Images (`.jpg`, `.png`) and Videos (`.mp4`, `.avi`).
* **Real-Time Face Extraction:** Uses MediaPipe to automatically locate, crop, and align faces before analysis.
* **Forensic Visualization:** Red heatmaps indicate high-confidence manipulated regions.
* **Cloud-Native:** Fully containerized and deployed on Hugging Face using `opencv-python-headless` for server stability.

---

## Technology Stack
* **Core Framework:** PyTorch
* **Computer Vision:** OpenCV, Albumentations, MediaPipe
* **Model Architecture:** EfficientNet, LSTM
* **Web Interface:** Gradio
* **Deployment:** Hugging Face Spaces (Debian Linux Environment)

---

## Statement on AI Usage
*As part of the academic transparency for this Capstone Project:*

**Generative AI (Gemini)** was utilized as a technical assistant during the development lifecycle. Specifically, it contributed to:
1.  **DevOps & Debugging:** Resolving Linux dependency conflicts (`libGL`/OpenCV) within the Hugging Face "headless" environment.
2.  **Architectural Consultation:** Providing comparative analysis between ResNet and EfficientNet to optimize for edge-deployment latency.
3.  **Documentation:** Assisting in the structuring of technical documentation and API guides.

*All core forensic logic, model training strategies, dataset curation, and final validation were designed, executed, and verified manually by the author.*

---

## Installation (Local)
To run this project locally:

1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/jerah-ello-dev/Project-Sentinel.git](https://github.com/jerah-ello-dev/Project-Sentinel.git)
    cd Project-Sentinel
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App:**
    ```bash
    python app.py
    ```
    *The Gradio interface will launch at http://127.0.0.1:7860*

---

## Author
**Jerah Meel Falcon Ello**
* Capstone Project - Postgraduate Diploma in Artificial Intellegence & Machine Learning
* Asian Institute of Management
