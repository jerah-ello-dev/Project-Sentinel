import gradio as gr
import torch
import cv2
import numpy as np
import mediapipe as mp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import your custom architectures
from model import SentinelModel
from model_rnn import TemporalSentinel
from gradcam import GradCAM, overlay_heatmap

# --- CONFIGURATION ---
# Hugging Face Free Tier is CPU-only. We must map location to cpu.
DEVICE = "cpu"
SEQUENCE_LENGTH = 10

print(f"🚀 Loading Sentinel Models on {DEVICE}...")

# 1. Load Spatial Model
spatial_model = SentinelModel().to(DEVICE)
spatial_checkpoint = torch.load("best_model.pth", map_location=torch.device('cpu'))
spatial_model.load_state_dict(spatial_checkpoint)
spatial_model.eval()

# 2. Load Temporal Model
temporal_model = TemporalSentinel(cnn_path=None).to(DEVICE)
temporal_checkpoint = torch.load("best_temporal_model.pth", map_location=torch.device('cpu'))
temporal_model.load_state_dict(temporal_checkpoint)
temporal_model.eval()

print("✅ Models Loaded Successfully!")

# --- PREDICTION LOGIC ---
def analyze_media(file_path):
    if file_path is None: 
        return None, "⚠️ Please upload a file."
    
    # Auto-detect file type
    if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
        return analyze_image(file_path)
    elif file_path.lower().endswith(('.mp4', '.avi', '.mov')):
        return analyze_video(file_path)
    else:
        return None, "❌ Unsupported file format. Please use JPG/PNG or MP4."

def analyze_image(path):
    print(f"🖼️ Analyzing Image: {path}")
    img = cv2.imread(path)
    if img is None: return None, "Error reading image."
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Face Detection
    mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    results = mp_face.process(img_rgb)
    
    if not results.detections:
        return img_rgb, "⚠️ No Face Detected. Cannot analyze."
    
    # Crop Face
    face = max(results.detections, key=lambda d: d.location_data.relative_bounding_box.width)
    bbox = face.location_data.relative_bounding_box
    h, w, _ = img.shape
    x, y, bw, bh = int(bbox.xmin*w), int(bbox.ymin*h), int(bbox.width*w), int(bbox.height*h)
    x, y = max(0, x), max(0, y)
    face_crop = img_rgb[y:y+bh, x:x+bw]
    
    if face_crop.size == 0: return img_rgb, "⚠️ Face too small or cropped out."

    # Preprocess
    transform = A.Compose([A.Resize(224, 224), A.Normalize(), ToTensorV2()])
    aug = transform(image=face_crop)['image'].unsqueeze(0).to(DEVICE)
    
    # Inference
    with torch.no_grad():
        score = spatial_model(aug).item()
    
    verdict_label = "🚨 DEEPFAKE DETECTED" if score > 0.50 else "✅ REAL IMAGE"
    conf_text = f"{verdict_label} (Confidence: {score*100:.2f}%)"
    
    # Explain (Grad-CAM)
    evidence_img = face_crop
    if score > 0.50:
        cam = GradCAM(model=spatial_model, target_layer=spatial_model.backbone.conv_head)
        heatmap = cam(aug)
        evidence_img = overlay_heatmap(heatmap, face_crop, alpha=0.5)
    
    return evidence_img, conf_text

def analyze_video(path):
    print(f"🎥 Analyzing Video: {path}")
    # Simplified Logic for Demo: Check the first valid sequence
    cap = cv2.VideoCapture(path)
    mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    transform = A.Compose([A.Resize(224, 224), A.Normalize(), ToTensorV2()])
    
    frame_buffer = []
    
    while cap.isOpened() and len(frame_buffer) < SEQUENCE_LENGTH:
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
    
    cap.release()
    
    if len(frame_buffer) < SEQUENCE_LENGTH:
        return None, "⚠️ Video too short or no face detected."
        
    seq = torch.stack(frame_buffer).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        score = temporal_model(seq).item()
        
    verdict_label = "🚨 DEEPFAKE VIDEO" if score > 0.50 else "✅ REAL VIDEO"
    return frame_buffer[-1].permute(1, 2, 0).cpu().numpy(), f"{verdict_label} (Confidence: {score*100:.2f}%)"

# --- UI SETUP ---
if __name__ == "__main__":
    interface = gr.Interface(
        fn=analyze_media,
        inputs=gr.File(label="Upload Image or Video"),
        outputs=[
            gr.Image(label="Forensic Evidence"), 
            gr.Label(label="Analysis Result")
        ],
        title="🛡️ Project Sentinel: Deepfake Forensics",
        description="Upload a media file. If it's a deepfake, the system will highlight the manipulated regions.",
        examples=[["best_model.pth"]] # Placeholder to prevent empty example error
    )
    interface.launch()