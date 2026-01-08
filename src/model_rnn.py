import torch
import torch.nn as nn
from model import SentinelModel

class TemporalSentinel(nn.Module):
    def __init__(self, cnn_path="./checkpoints/best_model.pth", hidden_dim=128, num_layers=2):
        super(TemporalSentinel, self).__init__()
        
        # 1. Load your Pre-Trained Spatial Model (The Eye)
        self.cnn = SentinelModel(pretrained=False) # Architecture only
        if cnn_path:
            self.cnn.load_state_dict(torch.load(cnn_path))
            print("✅ Loaded Pre-trained EfficientNet Weights")
            
        # Remove the last classification layer (Sigmoid)
        # We want the raw features (1280 dim for EfficientNet-B0)
        self.cnn.backbone.classifier = nn.Identity() 
        self.feature_dim = 1280 
        
        # Freeze CNN (Optional: Unfreeze if you have a massive GPU)
        for param in self.cnn.parameters():
            param.requires_grad = False
            
        # 2. The Temporal Head (The Brain) - LSTM
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # 3. Final Classifier
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Channels, Height, Width)
        batch_size, seq_len, c, h, w = x.size()
        
        # Pass each frame through CNN
        # Reshape to (Batch * Seq_Len, C, H, W) for faster processing
        c_in = x.view(batch_size * seq_len, c, h, w)
        features = self.cnn.backbone(c_in) # Output: (Batch * Seq, 1280)
        
        # Reshape back for LSTM: (Batch, Seq_Len, 1280)
        r_in = features.view(batch_size, seq_len, -1)
        
        # Pass through LSTM
        # out: (Batch, Seq_Len, Hidden), hidden: (Layers, Batch, Hidden)
        lstm_out, (h_n, c_n) = self.lstm(r_in)
        
        # We only care about the result of the LAST frame
        last_output = lstm_out[:, -1, :] 
        
        # Final Classification
        logits = self.fc(last_output)
        return self.sigmoid(logits).squeeze()