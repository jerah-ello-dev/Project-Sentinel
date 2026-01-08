import os
import cv2
import mediapipe as mp
import splitfolders
from tqdm import tqdm

# --- CONFIGURATION (Points to your folders) ---
PATH_RAW = "./data/raw"
PATH_PROCESSED = "./data/processed"
TEMP_OUTPUT = "./data/temp_faces"

# FRAMES TO SKIP (To avoid having 1 million identical images)
# 30 frames = approx 1 second. We take 1 photo every second.
FRAME_SKIP = 30 

# SETUP FACE DETECTOR (Google MediaPipe)
mp_face_detection = mp.solutions.face_detection
detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)

def extract_faces(video_path, label, output_folder):
    cap = cv2.VideoCapture(video_path)
    filename = os.path.basename(video_path).split('.')[0]
    count = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # Only take every 30th frame
        if count % FRAME_SKIP == 0:
            # Convert to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(image)
            
            # If face found, crop it
            if results.detections:
                for i, detection in enumerate(results.detections):
                    h, w, _ = frame.shape
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    bw = int(bbox.width * w)
                    bh = int(bbox.height * h)
                    
                    # Ensure crop is inside image
                    x, y = max(0, x), max(0, y)
                    
                    # Crop and Save
                    face = frame[y:y+bh, x:x+bw]
                    if face.size > 0:
                        save_name = f"{filename}_frame{count}.jpg"
                        save_path = os.path.join(output_folder, label, save_name)
                        cv2.imwrite(save_path, face)
        count += 1
    cap.release()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Create Folders
    os.makedirs(os.path.join(TEMP_OUTPUT, "real"), exist_ok=True)
    os.makedirs(os.path.join(TEMP_OUTPUT, "fake"), exist_ok=True)

    print("🚀 STARTING PREPROCESSING...")

    # 2. PROCESS FACE FORENSICS (FF++)
    # NOTE: You must check your specific FF++ folder name inside data/raw
    # Adjust 'FaceForensics' below to match your folder if different
    ff_path = os.path.join(PATH_RAW, "FaceForensics") 
    
    if os.path.exists(ff_path):
        print("Processing FaceForensics...")
        # Recursively find all mp4 files
        for root, dirs, files in os.walk(ff_path):
            for file in tqdm(files, desc="Scanning FF++"):
                if file.endswith(".mp4"):
                    full_path = os.path.join(root, file)
                    # Logic: If folder name contains 'original', it's REAL. Else FAKE.
                    if "original" in root:
                        extract_faces(full_path, "real", TEMP_OUTPUT)
                    else:
                        extract_faces(full_path, "fake", TEMP_OUTPUT)

    # 3. PROCESS CELEB-DF
    celeb_path = os.path.join(PATH_RAW, "Celeb-DF-v2")
    if os.path.exists(celeb_path):
        print("Processing Celeb-DF...")
        # Usually Celeb-real = Real, Celeb-synthesis = Fake
        for root, dirs, files in os.walk(celeb_path):
            for file in tqdm(files, desc="Scanning Celeb-DF"):
                if file.endswith(".mp4"):
                    full_path = os.path.join(root, file)
                    if "Celeb-real" in root:
                        extract_faces(full_path, "real", TEMP_OUTPUT)
                    elif "Celeb-synthesis" in root:
                        extract_faces(full_path, "fake", TEMP_OUTPUT)

    # 4. PROCESS GENIMAGE (Already Images)
    # We will simply copy them or run face detection if needed.
    # For now, let's assume they are ready to go.
    gen_path = os.path.join(PATH_RAW, "GenImage")
    # (Add GenImage logic here if needed, but let's test video first)

    # 5. SPLIT INTO TRAIN / VAL
    print("✂️ Splitting into Train/Val folders...")
    splitfolders.ratio(TEMP_OUTPUT, output=PATH_PROCESSED, seed=1337, ratio=(.8, .2), group_prefix=None, move=False)
    
    print(f"✅ DONE! Data is ready in {PATH_PROCESSED}")