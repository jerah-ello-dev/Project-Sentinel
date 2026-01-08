import os
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        """
        root_dir: Path to 'data/processed' (e.g. ./data/processed)
        split: 'train' or 'val'
        """
        # Construct the full path (e.g., ./data/processed/train)
        self.split_dir = os.path.join(root_dir, split)
        
        # DEBUG CHECK: Does the folder exist?
        if not os.path.exists(self.split_dir):
            raise FileNotFoundError(f"❌ CRITICAL ERROR: The folder '{self.split_dir}' does not exist.\nDid you run preprocess.py? Is your folder named 'processed' or something else?")

        # 1. Gather all file paths
        # We assume structure: data/processed/train/real and data/processed/train/fake
        real_folder = os.path.join(self.split_dir, "real")
        fake_folder = os.path.join(self.split_dir, "fake")

        self.real_paths = self._get_files(real_folder)
        self.fake_paths = self._get_files(fake_folder)
        
        # 2. Create labels (0 = Real, 1 = Fake)
        self.all_paths = self.real_paths + self.fake_paths
        self.labels = [0] * len(self.real_paths) + [1] * len(self.fake_paths)

        # DEBUG CHECK: Did we find any images?
        if len(self.all_paths) == 0:
             print(f"⚠️ WARNING: No images found in {self.split_dir}. Check your folders!")

        # 3. Define Augmentations
        if split == "train":
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussNoise(p=0.2),
                A.Normalize(),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(),
                ToTensorV2()
            ])

    def _get_files(self, folder):
        files = []
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    files.append(os.path.join(folder, f))
        return files

    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, idx):
        path = self.all_paths[idx]
        label = self.labels[idx]
        
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"❌ Error: Could not read image at {path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        augmented = self.transform(image=image)
        image_tensor = augmented['image']
        
        return image_tensor, torch.tensor(label, dtype=torch.float32)

# --- TEST BLOCK (MUST BE UNINDENTED - TOUCHING THE LEFT SIDE) ---
if __name__ == "__main__":
    print("🔍 Testing Dataset Loading...")
    
    # POINT THIS TO YOUR EXACT 'data/processed' FOLDER
    # Based on your folder structure, it should be relative to where you run the command
    dataset_root = "./data/processed" 
    
    try:
        # Try loading the 'train' split
        dataset = DeepfakeDataset(root_dir=dataset_root, split="train")
        
        print(f"✅ SUCCESS: Dataset Loaded!")
        print(f"📂 Found {len(dataset)} total images.")
        print(f"   - Real images: {len(dataset.real_paths)}")
        print(f"   - Fake images: {len(dataset.fake_paths)}")
        
        if len(dataset) > 0:
            img, label = dataset[0]
            print(f"🖼️ Sample Image Shape: {img.shape}")
            print(f"🏷️ Sample Label: {label}")
        else:
            print("❌ ERROR: Dataset is empty. Check data/processed/train/real folder.")
            
    except Exception as e:
        print(f"\n❌ CRASHED: {e}")