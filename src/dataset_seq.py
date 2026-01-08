import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SequenceDataset(Dataset):
    def __init__(self, root_dir, split="train", sequence_length=10):
        """
        Loads sequences of frames (e.g. 10 frames from the same video)
        """
        self.split_dir = os.path.join(root_dir, split)
        self.seq_len = sequence_length
        
        # 1. Group images by Video ID
        self.real_videos = self._group_by_video(os.path.join(self.split_dir, "real"))
        self.fake_videos = self._group_by_video(os.path.join(self.split_dir, "fake"))
        
        self.all_videos = self.real_videos + self.fake_videos
        self.labels = [0] * len(self.real_videos) + [1] * len(self.fake_videos)

        # Transforms (Resize only, no random flips to keep motion consistent)
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(),
            ToTensorV2()
        ])

    def _group_by_video(self, folder):
        """
        Groups frames like 'vid1_frame1.jpg', 'vid1_frame2.jpg' into lists.
        """
        if not os.path.exists(folder): return []
        
        video_dict = {}
        for f in os.listdir(folder):
            if f.endswith('.jpg'):
                # Assumes filename format: "videoname_frameX.jpg"
                # We split by '_' and take everything except the last part as the ID
                video_id = "_".join(f.split("_")[:-1]) 
                path = os.path.join(folder, f)
                
                if video_id not in video_dict:
                    video_dict[video_id] = []
                video_dict[video_id].append(path)
        
        # Filter out videos that are too short
        valid_sequences = []
        for vid in video_dict.values():
            # Sort by frame number (important for temporal logic!)
            vid.sort(key=lambda x: int(x.split("frame")[-1].split(".")[0]))
            
            # Create sliding windows (chunks of 10 frames)
            for i in range(0, len(vid) - self.seq_len, self.seq_len):
                valid_sequences.append(vid[i : i + self.seq_len])
                
        return valid_sequences

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        frame_paths = self.all_videos[idx]
        label = self.labels[idx]
        
        frames_tensor = []
        for path in frame_paths:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transform(image=img)['image']
            frames_tensor.append(img)
            
        # Stack frames: (Sequence_Len, Channels, Height, Width)
        return torch.stack(frames_tensor), torch.tensor(label, dtype=torch.float32)