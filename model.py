import torch
import torch.nn as nn
import timm # Library for state-of-the-art models

class SentinelModel(nn.Module):
    def __init__(self, model_name='efficientnet_b0', pretrained=True):
        super(SentinelModel, self).__init__()
        
        # 1. Load the Backbone (EfficientNet)
        # num_classes=1 because we output a single probability (Real vs Fake)
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=1)
        
        # 2. Add a Sigmoid activation to force output between 0 and 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pass image through the backbone
        logits = self.backbone(x)
        # Convert to probability
        probs = self.sigmoid(logits)
        # Remove extra dimension (Batch, 1) -> (Batch)
        return probs.squeeze()

# --- TEST BLOCK ---
if __name__ == "__main__":
    # Create a dummy image (Batch=4, Color=3, Size=224x224)
    dummy_input = torch.randn(4, 3, 224, 224)
    
    # Initialize Model
    model = SentinelModel()
    
    # Run Inference
    output = model(dummy_input)
    print(f"✅ Model Built Successfully!")
    print(f"Output Shape: {output.shape} (Should be torch.Size([4]))")
    print(f"Sample Scores: {output}")