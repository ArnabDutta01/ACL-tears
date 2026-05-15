# ============================================================
# V9 Single Image Prediction + GradCAM (Local Laptop)
# ============================================================
# Load a PNG MRI slice, predict ACL tear, and show GradCAM heatmap

# %% [markdown]
# # Cell 1: Configuration

# %%
import os
MODEL_PATH = r'c:\ML projects\ACL tears\V9\best_acl_model_v9.pth'
# --- Change this to your PNG path ---
IMAGE_PATH = r'c:\ML projects\ACL tears\DATASET\test_image.png'
TARGET_SIZE = (256, 256)  # V9 was trained on 256x256

# %% [markdown]
# # Cell 2: Imports

# %%
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# %% [markdown]
# # Cell 3: Model (must match V9 training architecture)

# %%
class MRNetV9(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        backbone = models.efficientnet_b0(weights=None)
        self.features = backbone.features
        self.pool = backbone.avgpool
        self.drop = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(1280, 2)

    def forward(self, x):
        x = x.squeeze(0)  # (S, 3, H, W)
        features = self.features(x)
        pooled = self.pool(features)
        pooled = pooled.flatten(1)
        volume_feat = torch.max(pooled, 0, keepdim=True)[0]
        volume_feat = self.drop(volume_feat)
        output = self.classifier(volume_feat)
        return output

# %% [markdown]
# # Cell 4: GradCAM Helper

# %%
class GradCAM:
    """GradCAM for the V9 EfficientNet model."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=1):
        """Generate GradCAM heatmap for the given class (1=Tear by default)."""
        self.model.eval()
        # Need gradients for GradCAM
        input_tensor.requires_grad_(True)

        # Forward pass (bypass squeeze since we handle it manually)
        x = input_tensor.squeeze(0)  # (S, 3, H, W)
        features = self.model.features(x)
        pooled = self.model.pool(features)
        pooled = pooled.flatten(1)
        volume_feat = torch.max(pooled, 0, keepdim=True)[0]
        output = self.model.classifier(volume_feat)

        # Backward pass for target class
        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        # Compute GradCAM
        gradients = self.gradients  # (S, C, H, W)
        activations = self.activations  # (S, C, H, W)

        # Global average pooling of gradients
        weights = gradients.mean(dim=[2, 3], keepdim=True)  # (S, C, 1, 1)
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (S, 1, H, W)
        cam = F.relu(cam)  # Only positive contributions

        # For single slice, take the first (and only) slice
        # For multiple slices, take the max-contributing slice
        cam_per_slice = cam.squeeze(1).sum(dim=[1, 2])  # (S,)
        best_slice_idx = cam_per_slice.argmax().item()

        best_cam = cam[best_slice_idx, 0].cpu().numpy()
        # Normalize to [0, 1]
        if best_cam.max() > 0:
            best_cam = (best_cam - best_cam.min()) / (best_cam.max() - best_cam.min())

        probs = torch.softmax(output, dim=1)
        return best_cam, probs, best_slice_idx

# %% [markdown]
# # Cell 5: Load Model

# %%
print("Loading V9 model...")
model = MRNetV9()
state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
print("  Model loaded successfully")

# Set up GradCAM on last conv layer of EfficientNet
target_layer = model.features[-1]  # Last block of EfficientNet features
gradcam = GradCAM(model, target_layer)

# %% [markdown]
# # Cell 6: Load & Preprocess Image

# %%
def load_png(image_path, target_size=(256, 256)):
    """Load a PNG MRI image and preprocess for V9 model."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    print(f"  Original size: {img.shape}")
    print(f"  Pixel range: [{img.min()}, {img.max()}]")

    # Resize to training dimensions
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

    # Normalize to [0, 1] — matching V9 training
    img_float = img.astype(np.float32) / 255.0

    # Stack to 3 channels, add slice and batch dims: (1, 1, 3, H, W)
    img_3ch = np.stack((img_float,)*3, axis=0)  # (3, H, W)
    tensor = torch.FloatTensor(img_3ch).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, H, W)

    return tensor, img_float

print(f"Loading image: {IMAGE_PATH}")
input_tensor, original_img = load_png(IMAGE_PATH, TARGET_SIZE)
input_tensor = input_tensor.to(device)
print(f"  Tensor shape: {input_tensor.shape}")

# %% [markdown]
# # Cell 7: Predict + GradCAM

# %%
# Run GradCAM (also gets prediction)
heatmap, probs, best_slice = gradcam.generate(input_tensor, target_class=1)

prob_normal = probs[0][0].item()
prob_tear = probs[0][1].item()
prediction = 'TEAR' if prob_tear >= 0.5 else 'NORMAL'
confidence = max(prob_normal, prob_tear) * 100

print('=' * 50)
print(f'PREDICTION: {prediction}')
print(f'  P(Normal) = {prob_normal:.4f} ({prob_normal*100:.1f}%)')
print(f'  P(Tear)   = {prob_tear:.4f} ({prob_tear*100:.1f}%)')
print(f'  Confidence: {confidence:.1f}%')
print('=' * 50)

# %% [markdown]
# # Cell 8: Visualize GradCAM

# %%
# Resize heatmap to match original image
heatmap_resized = cv2.resize(heatmap, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)

# Create colored heatmap overlay
heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]  # RGB from jet colormap
heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

# Create overlay
original_rgb = np.stack([original_img]*3, axis=-1)  # Grayscale to RGB
overlay = 0.6 * original_rgb + 0.4 * heatmap_colored / 255.0
overlay = np.clip(overlay, 0, 1)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Original
axes[0].imshow(original_img, cmap='gray')
axes[0].set_title('Original MRI Slice', fontsize=14)
axes[0].axis('off')

# Heatmap
im = axes[1].imshow(heatmap_resized, cmap='jet', vmin=0, vmax=1)
axes[1].set_title('GradCAM Heatmap', fontsize=14)
axes[1].axis('off')
plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

# Overlay
axes[2].imshow(overlay)
axes[2].set_title(f'Overlay — {prediction} ({confidence:.1f}%)', fontsize=14,
                  color='red' if prediction == 'TEAR' else 'green', fontweight='bold')
axes[2].axis('off')

plt.suptitle(f'V9 ACL Tear Detection — P(Tear)={prob_tear:.3f}', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(r'c:\ML projects\ACL tears\V9\gradcam_output.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\nSaved to: V9/gradcam_output.png")

# %% [markdown]
# # Cell 9: Batch Predict on Multiple PNGs (Optional)

# %%
def predict_png(image_path, model, gradcam, device, target_size=(256, 256)):
    """Predict a single PNG and return results."""
    tensor, img = load_png(image_path, target_size)
    tensor = tensor.to(device)
    heatmap, probs, _ = gradcam.generate(tensor, target_class=1)
    prob_tear = probs[0][1].item()
    pred = 'TEAR' if prob_tear >= 0.5 else 'NORMAL'
    return pred, prob_tear, img, heatmap

# Example: predict all PNGs in a folder
# Uncomment and set your folder path:
# PNG_FOLDER = r'c:\ML projects\ACL tears\test_pngs'
# for png_file in sorted(Path(PNG_FOLDER).glob('*.png')):
#     pred, prob, _, _ = predict_png(png_file, model, gradcam, device)
#     print(f"{png_file.name}: {pred} (P(Tear)={prob:.4f})")
