import torch
import torch.nn.functional as F
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook into the model to catch the "thoughts" (gradients)
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x):
        # 1. Forward Pass
        output = self.model(x)
        
        # 2. Backward Pass (Force the AI to explain "Why is this Fake?")
        self.model.zero_grad()
        score = output  # The probability score
        score.backward() # Trace it back to the pixels

        # 3. Generate Heatmap
        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]
        
        # Weight the channels by their gradients (Importance)
        weights = np.mean(gradients, axis=(1, 2))
        for i, w in enumerate(weights):
            activations[i, :, :] *= w
            
        # Average the channels to get a 2D map
        heatmap = np.mean(activations, axis=0)
        heatmap = np.maximum(heatmap, 0) # ReLU (We only care about positive activation)
        
        # Normalize between 0 and 1
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
        return heatmap

# Original code for overlay_heatmap function
def overlay_heatmap(heatmap, original_image):
    # Resize heatmap to match original image size
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    
    # Convert to Red-Blue color map
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Blend: 40% Heatmap + 60% Original Image
    superimposed = cv2.addWeighted(heatmap, 0.4, original_image, 0.6, 0)
    return superimposed

# revired overlay_heatmap function to fix color convesion issue
#def overlay_heatmap(heatmap, original_image):
    # Resize heatmap to match original image size
#    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    
    # --- THE SWAP: Invert the heatmap values ---
    # 0.0 becomes 1.0 (Red), 1.0 becomes 0.0 (Blue)
#    inverted_heatmap = 1.0 - heatmap
    
    # Convert to Red-Blue color map using the INVERTED values
#    heatmap_uint8 = np.uint8(255 * inverted_heatmap)
#    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # Blend: 40% Heatmap + 60% Original Image
#    superimposed = cv2.addWeighted(heatmap_colored, 0.4, original_image, 0.6, 0)
#    return superimposed

