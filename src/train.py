import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# IMPORT YOUR MODULES
from dataset import DeepfakeDataset
from model import SentinelModel

# --- CONFIGURATION (The "Knobs" you can turn) ---
BATCH_SIZE = 16          # How many images to learn from at once (Lower if PC freezes)
LEARNING_RATE = 0.001    # How fast the model learns (Too fast = unstable)
EPOCHS = 10              # How many times to read the entire dataset
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "./checkpoints"

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train() # Set to "Training Mode" (Enable Dropout/Batchnorm)
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Progress Bar
    loop = tqdm(loader, desc="Training", leave=False)
    
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        
        # 1. Forward Pass (Make a guess)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 2. Backward Pass (Calculate errors)
        optimizer.zero_grad() # Reset gradients
        loss.backward()       # Calculate new gradients
        optimizer.step()      # Update weights
        
        # 3. Track Stats
        running_loss += loss.item()
        predicted = (outputs > 0.5).float() # If prob > 0.5, it's Fake (1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update Progress Bar
        loop.set_postfix(loss=loss.item())
        
    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def validate(model, loader, criterion, device):
    model.eval() # Set to "Evaluation Mode" (Disable Dropout)
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad(): # Don't calculate gradients (saves memory)
        for images, labels in tqdm(loader, desc="🧪 Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# MAIN EXECUTION
if __name__ == "__main__":
    # 0. Setup
    os.makedirs(SAVE_PATH, exist_ok=True)
    print(f"Device selected: {DEVICE}")
    if DEVICE == 'cpu':
        print("WARNING: Training on CPU will be slow. Use a GPU if possible!")

    # 1. Load Data
    print("Loading Datasets...")
    train_ds = DeepfakeDataset(root_dir="./data/processed", split="train")
    val_ds = DeepfakeDataset(root_dir="./data/processed", split="val")
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Data Loaded. Train: {len(train_ds)}, Val: {len(val_ds)}")

    # 2. Load Model
    print("Initializing Model...")
    model = SentinelModel().to(DEVICE)
    
    # 3. Setup Optimizer & Loss
    criterion = nn.BCELoss() # Binary Cross Entropy (For Real vs Fake)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 4. Training Loop
    best_acc = 0.0
    
    print(f"\nSTARTING TRAINING FOR {EPOCHS} EPOCHS\n")
    
    for epoch in range(EPOCHS):
        print(f"--- Epoch {epoch+1}/{EPOCHS} ---")
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        print(f"   Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Acc: {val_acc:.2f}%")
        
        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(SAVE_PATH, "best_model.pth"))
            print(f"   New Best Model Saved! ({best_acc:.2f}%)")
            
    print("\nTRAINING COMPLETE!")