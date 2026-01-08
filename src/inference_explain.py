import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse
import os
import numpy as np
import mediapipe as mp
from collections import deque
import matplotlib.pyplot as plt

from model import SentinelModel
from model_rnn import TemporalSentinel
from gradcam import GradCAM, overlay_heatmap

# --- CONFIGURATION ---
TEMPORAL_PATH = "./checkpoints/best_temporal_model.pth"
SPATIAL_PATH = "./checkpoints/best_model.pth" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQUENCE_LENGTH = 10

def explain_video(video_path):
    print(f"🔍 Forensic Analysis: {video_path}")
    
    # 1. Load Temporal Model (To find the Glitch)
    if not os.path.exists(TEMPORAL_PATH): 
        print("❌ Error: best_temporal_model.pth not found!")
        return
        
    temp_model = TemporalSentinel(cnn_path=None).to(DEVICE)
    temp_model.load_state_dict(torch.load(TEMPORAL_PATH))
    temp_model.eval()
    
    # 2. Load Spatial Model (To Explain the Glitch)
    if not os.path.exists(SPATIAL_PATH):
        print("❌ Error: best_model.pth not found!")
        return
        
    spatial_model = SentinelModel().to(DEVICE)
    spatial_model.load_state_dict(torch.load(SPATIAL_PATH))
    spatial_model.eval() 
    
    # HOOK GRAD-CAM
    target_layer = spatial_model.backbone.conv_head 
    cam = GradCAM(model=spatial_model, target_layer=target_layer)

    # 3. Setup Video & Face Detector
    cap = cv2.VideoCapture(video_path)
    
    # --- THE FIX IS HERE ---
    mp_face = mp.solutions.face_detection.FaceDetection(
        model_selection=1, 
        min_detection_confidence=0.5  # <--- FIXED TYPO HERE
    )
    
    transform = A.Compose([A.Resize(224, 224), A.Normalize(), ToTensorV2()])
    
    frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
    raw_face_buffer = deque(maxlen=SEQUENCE_LENGTH) 
    
    highest_score = 0
    worst_frame_face = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # --- PHASE 1: FIND THE FAKE ---
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face.process(img_rgb)
        
        if results.detections:
            face = max(results.detections, key=lambda d: d.location_data.relative_bounding_box.width)
            bbox = face.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, bw, bh = int(bbox.xmin*w), int(bbox.ymin*h), int(bbox.width*w), int(bbox.height*h)
            x, y = max(0, x), max(0, y)
            
            face_crop = img_rgb[y:y+bh, x:x+bw]
            if face_crop.size > 0:
                # Add to buffer
                aug = transform(image=face_crop)['image']
                frame_buffer.append(aug)
                raw_face_buffer.append(face_crop) 
                
                if len(frame_buffer) == SEQUENCE_LENGTH:
                    seq = torch.stack(list(frame_buffer)).unsqueeze(0).to(DEVICE)
                    
                    with torch.no_grad():
                        score = temp_model(seq).item()
                        
                    # Track the "Worst" frame (Highest Suspicion)
                    if score > highest_score:
                        highest_score = score
                        worst_frame_face = raw_face_buffer[-1].copy()
                        
    cap.release()
    
    print("="*40)
    print(f"PEAK SUSPICION SCORE: {highest_score*100:.2f}%")
    
    # --- PHASE 2: EXPLAIN THE FAKE ---
    # We use a lower threshold (0.50) to ensure we generate evidence if it's suspicious at all
    if highest_score > 0.50 and worst_frame_face is not None:
        print("Generating Evidence Heatmap...")
        
        aug_face = transform(image=worst_frame_face)['image'].unsqueeze(0).to(DEVICE)
        
        spatial_model.zero_grad()
        heatmap = cam(aug_face)
        result_img = overlay_heatmap(heatmap, worst_frame_face)
        
        output_filename = "evidence_deepfake.jpg"
        cv2.imwrite(output_filename, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        print(f"Evidence saved to: {output_filename}")
        print("   (Red areas indicate the pixels that triggered the detection)")
    else:
        print("Video appears Real (No high-confidence glitch found).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    args = parser.parse_args()
    explain_video(args.video)