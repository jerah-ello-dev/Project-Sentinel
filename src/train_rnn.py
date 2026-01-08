import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from dataset_seq import SequenceDataset
from model_rnn import TemporalSentinel

# --- CONFIGURATION ---
BATCH_SIZE = 4 
EPOCHS = 10
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "./checkpoints"

def train():
    # 1. Load Data
    print("📂 Loading Sequence Data (Grouping frames)...")
    # Note: Ensure preprocess.py was run so 'data/processed' exists
    train_ds = SequenceDataset(root_dir="./data/processed", split="train", sequence_length=10)
    val_ds = SequenceDataset(root_dir="./data/processed", split="val", sequence_length=10)
    
    # Drop_last=True prevents the "Batch=1" crash if dataset size isn't divisible by 4
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    
    # 2. Load Model
    print("🧠 Initializing Temporal Model...")
    if not os.path.exists("./checkpoints/best_model.pth"):
        print("❌ CRITICAL: ./checkpoints/best_model.pth not found!")
        print("   Please run 'train.py' (Spatial Training) first.")
        return

    model = TemporalSentinel(cnn_path="./checkpoints/best_model.pth").to(DEVICE)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    best_acc = 0.0
    
    # 3. Training Loop
    print(f"\n🔥 STARTING TEMPORAL TRAINING (LSTM)\n")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for sequences, labels in loop:
            sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
            
            # Forward Pass
            outputs = model(sequences)
            
            # --- THE FIX IS HERE ---
            # Force both to be 1D vectors to avoid Shape Mismatch
            outputs = outputs.view(-1)
            labels = labels.view(-1)
            
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loop.set_postfix(loss=loss.item())
            
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for seq, lbl in val_loader:
                seq, lbl = seq.to(DEVICE), lbl.to(DEVICE)
                out = model(seq).view(-1)
                lbl = lbl.view(-1)
                
                pred = (out > 0.5).float()
                val_total += lbl.size(0)
                val_correct += (pred == lbl).sum().item()
        
        # Avoid division by zero if validation set is empty
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        
        print(f"   📉 Train Acc: {100*correct/total:.2f}% | 🧪 Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(SAVE_PATH, "best_temporal_model.pth"))
            print("   💾 Saved Best Temporal Model")

if __name__ == "__main__":
    train()