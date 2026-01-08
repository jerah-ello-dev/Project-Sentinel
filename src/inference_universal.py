import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse
import os
import numpy as np
import mediapipe as mp
from collections import deque

from model import SentinelModel
from model_rnn import TemporalSentinel
from gradcam import GradCAM, overlay_heatmap

# --- CONFIGURATION ---
TEMPORAL_PATH = "./checkpoints/best_temporal_model.pth"
SPATIAL_PATH = "./checkpoints/best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQUENCE_LENGTH = 10

# --- SETUP MODELS ---
def load_models():
    # Load Spatial (For Images & Explanation)
    spatial = SentinelModel().to(DEVICE)
    if os.path.exists(SPATIAL_PATH):
        spatial.load_state_dict(torch.load(SPATIAL_PATH))
    spatial.eval()
    
    # Load Temporal (For Video)
    temporal = None
    if os.path.exists(TEMPORAL_PATH):
        temporal = TemporalSentinel(cnn_path=None).to(DEVICE)
        temporal.load_state_dict(torch.load(TEMPORAL_PATH))
        temporal.eval()
        
    return spatial, temporal

def analyze_image(path, spatial_model):
    print(f"🖼️  Analyzing IMAGE: {path}")
    
    # 1. Load & Detect Face
    img = cv2.imread(path)
    if img is None:
        print("❌ Error: Could not read image.")
        return

    mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp_face.process(img_rgb)
    
    if not results.detections:
        print("⚠️ No face detected in image.")
        return

    # 2. Crop Face
    face = max(results.detections, key=lambda d: d.location_data.relative_bounding_box.width)
    bbox = face.location_data.relative_bounding_box
    h, w, _ = img.shape
    x, y, bw, bh = int(bbox.xmin*w), int(bbox.ymin*h), int(bbox.width*w), int(bbox.height*h)
    x, y = max(0, x), max(0, y)
    face_crop = img_rgb[y:y+bh, x:x+bw]
    
    # 3. Predict (Spatial Only)
    transform = A.Compose([A.Resize(224, 224), A.Normalize(), ToTensorV2()])
    aug = transform(image=face_crop)['image'].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        score = spatial_model(aug).item()
        
    print("="*40)
    print(f"📊 SUSPICION SCORE: {score*100:.2f}%")
    
    # 4. The Verdict & Explanation
    if score > 0.50:
        print("🚨 VERDICT: AI-GENERATED IMAGE DETECTED") # <--- ADDED VERDICT
        print("📸 Generating Evidence Heatmap...")
        
        target_layer = spatial_model.backbone.conv_head
        cam = GradCAM(model=spatial_model, target_layer=target_layer)
        
        spatial_model.zero_grad()
        heatmap = cam(aug)
        result_img = overlay_heatmap(heatmap, face_crop)
        
        cv2.imwrite("evidence_image.jpg", cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        print(f"✅ Evidence saved to: evidence_image.jpg")
    else:
        print("✅ VERDICT: REAL IMAGE") # <--- ADDED VERDICT

def analyze_video(path, spatial_model, temporal_model):
    print(f"🎥 Analyzing VIDEO: {path}")
    
    cap = cv2.VideoCapture(path)
    mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    transform = A.Compose([A.Resize(224, 224), A.Normalize(), ToTensorV2()])
    
    frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
    raw_face_buffer = deque(maxlen=SEQUENCE_LENGTH)
    highest_score = 0
    worst_frame_face = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
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
                aug = transform(image=face_crop)['image']
                frame_buffer.append(aug)
                raw_face_buffer.append(face_crop)
                
                if len(frame_buffer) == SEQUENCE_LENGTH:
                    seq = torch.stack(list(frame_buffer)).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        score = temporal_model(seq).item()
                    if score > highest_score:
                        highest_score = score
                        worst_frame_face = raw_face_buffer[-1].copy()
    cap.release()
    
    print("="*40)
    print(f"PEAK VIDEO SUSPICION: {highest_score*100:.2f}%")
    
    if highest_score > 0.50:
        print("VERDICT: DEEPFAKE VIDEO DETECTED") # <--- ADDED VERDICT
        if worst_frame_face is not None:
            print("Generating Evidence Heatmap...")
            target_layer = spatial_model.backbone.conv_head
            cam = GradCAM(model=spatial_model, target_layer=target_layer)
            
            aug = transform(image=worst_frame_face)['image'].unsqueeze(0).to(DEVICE)
            spatial_model.zero_grad()
            heatmap = cam(aug)
            result_img = overlay_heatmap(heatmap, worst_frame_face)
            
            cv2.imwrite("evidence_video.jpg", cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
            print(f"Evidence saved to: evidence_video.jpg")
    else:
        print("VERDICT: REAL VIDEO") # <--- ADDED VERDICT

# --- MAIN ROUTER ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to image or video")
    args = parser.parse_args()
    
    spatial, temporal = load_models()
    
    # Auto-detect file type
    if args.file.lower().endswith(('.mp4', '.avi', '.mov')):
        if temporal:
            analyze_video(args.file, spatial, temporal)
        else:
            print("Error: Video detected but Temporal Model not found.")
    elif args.file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
        analyze_image(args.file, spatial)
    else:
        print("Unknown file type. Use .mp4 or .jpg")