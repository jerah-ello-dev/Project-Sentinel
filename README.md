# Project Sentinel

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![License](https://img.shields.io/badge/License-Academic-green)
![Status](https://img.shields.io/badge/Status-Capstone%20Project-success)

**Project Sentinel** is an automated deepfake detection and forensic analysis system developed as a Capstone Project at the **Asian Institute of Management**.

In an era where the "Liar's Dividend" allows authentic media to be dismissed as fake, Sentinel provides a robust, evidence-based solution. Unlike traditional "black box" detectors, this system employs a **Hybrid Sequential Architecture (CNN + LSTM)** to analyze temporal inconsistencies and integrates **Explainable AI (Grad-CAM)** to provide visual forensic proof of manipulation.

---

## Table of Contents
- [Key Features](#-key-features)
- [Technical Architecture](#-technical-architecture)
- [Performance](#-performance)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Limitations & Future Work](#-limitations--future-work)

---

## Key Features

* **Sequential Analysis:** Utilizes an LSTM network to detect motion-based anomalies (e.g., unnatural blinking, lip-sync jitters) that single-frame detectors miss.
* **Forensic Explainability:** Integrated **Grad-CAM** generates heatmaps to visualize exactly *where* manipulation was detected, transforming the output from a simple score to auditable evidence.
* **Universal Inference:** A streamlined pipeline (`inference_universal.py`) capable of processing both static images and video streams.
* **Robust Preprocessing:** Includes face gating via Google MediaPipe and standard ImageNet normalization to ensure high-quality input.

---

## Technical Architecture

The system is built on a two-stream hybrid architecture:

1.  **Spatial "Eye" (CNN Backbone):**
    * **Model:** EfficientNet-B0 (Pre-trained on ImageNet).
    * **Role:** Extracts high-level spatial features (textures, edges, blending artifacts) from individual frames. EfficientNet was chosen for its parameter efficiency (10x fewer parameters than ResNet-50).

2.  **Temporal "Brain" (RNN/LSTM):**
    * **Model:** Long Short-Term Memory (LSTM) Network.
    * **Role:** Analyzes a sliding window of 10 consecutive feature vectors to identify temporal glitches and inconsistencies across time.

---

## Performance

The models were rigorously evaluated on a diverse dataset combination of **FaceForensics++**, **Celeb-DF v2**, and **Tiny GenImage**.

| Metric | Spatial Model (CNN Only) | Temporal Model (CNN + LSTM) |
| :--- | :--- | :--- |
| **Validation Accuracy** | 94.35% | **95.83%** |
| **Validation Loss** | 0.1356 | **0.0027** |
| **Inference Focus** | Texture & Blending Artifacts | Motion & Temporal Consistency |

*Data Source: Project Sentinel Capstone Report.*

---

## Repository Structure

The project follows a modular production-ready structure:

```text
Project_Sentinel/
├── checkpoints/                 # Saved model weights (best_model.pth)
├── data/                        # Data storage (processed/raw)
├── src/                         # Source Code
│   ├── inference_universal.py   # MAIN SCRIPT: Universal entry point
│   ├── model.py                 # CNN Backbone architecture
│   ├── model_rnn.py             # LSTM Temporal architecture
│   ├── gradcam.py               # Explainable AI (Heatmap generation)
│   ├── preprocess.py            # Data pipeline & augmentation
│   ├── dataset_seq.py           # Sequential data loader
│   ├── train.py                 # Spatial training loop
│   └── train_rnn.py             # Temporal training loop
├── .gitignore                   # Git configuration
├── requirements.txt             # Dependency list
└── README.md                    # Project documentation

```

---

## Installation

### 1. Prerequisites

* Python 3.8+
* CUDA-enabled GPU (Recommended for real-time inference)

### 2. Clone the Repository

```bash
git clone [https://github.com/jerah-ello-dev/Project-Sentinel.git](https://github.com/jerah-ello-dev/Project-Sentinel.git)
cd Project-Sentinel

```

### 3. Install Dependencies

It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt

```

### 4. Model Weights

Ensure your trained weights (e.g., `best_model.pth` or `best_temporal_model.pth`) are placed in the `checkpoints/` directory.

---

## Usage

To run the system, use the **Universal Inference** script. This script automatically handles model initialization, data transformation, and visualization.

```bash
python src/inference_universal.py

```

**What to expect:**

1. The system initializes the EfficientNet-LSTM architecture.
2. It opens a video interface (webcam or file input).
3. Real-time **Grad-CAM heatmaps** will overlay on the video, highlighting suspicious regions.
4. A confidence score and label (REAL/FAKE) will be displayed.

---

## Limitations & Future Work

**Current Limitations:**

 
**Demographic Bias:** The primary training datasets are skewed towards Caucasian subjects, which may impact accuracy on under-represented groups.


* **Lighting Sensitivity:** Performance may degrade in low-light or over-exposed environments.


**Hardware Latency:** Real-time XAI generation is computationally expensive on non-GPU devices.



**Future Roadmap:**

 
**Dataset Localization:** Retraining on a localized dataset featuring Filipino subjects to mitigate racial bias.


 
**Adversarial Training:** Hardening the model against anti-forensic attacks (e.g., noise injection, blurring).


* **Edge Deployment:** Optimization for deployment on mobile or edge devices.

---

## License & Credits

**Author:** Jerah Meel Falcon Ello 

**Institution:** Asian Institute of Management (AIM) 

This project is submitted in partial fulfillment of the requirements for the **Postgraduate Diploma in Artificial Intelligence and Machine Learning**.

```

```