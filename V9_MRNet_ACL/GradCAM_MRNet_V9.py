# ============================================================
# V9 GradCAM Visualization on MRNet Dataset (20 samples)
# ============================================================

# %% [markdown]
# # Cell 1: Configuration

# %%
import os
MODEL_PATH = r'c:\ML projects\ACL tears\V9\best_acl_model_v9.pth'
DATA_DIR = r'c:\ML projects\ACL tears\DATASET\mrnet_sagittal'
NUM_SAMPLES = 20  # 10 Normal + 10 Tear
MAX_SLICES = 40
SEED = 42

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
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
np.random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# %% [markdown]
# # Cell 3: Model + GradCAM

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
        x = x.squeeze(0)
        features = self.features(x)
        pooled = self.pool(features)
        pooled = pooled.flatten(1)
        volume_feat = torch.max(pooled, 0, keepdim=True)[0]
        volume_feat = self.drop(volume_feat)
        output = self.classifier(volume_feat)
        return output


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_act)
        target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, m, i, o):
        self.activations = o.detach()

    def _save_grad(self, m, gi, go):
        self.gradients = go[0].detach()

    def generate(self, input_tensor, target_class=1):
        self.model.eval()
        input_tensor.requires_grad_(True)
        x = input_tensor.squeeze(0)
        features = self.model.features(x)
        pooled = self.model.pool(features)
        pooled = pooled.flatten(1)
        volume_feat = torch.max(pooled, 0, keepdim=True)[0]
        output = self.model.classifier(volume_feat)

        self.model.zero_grad()
        output[0, target_class].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Per-slice cam strength
        cam_strength = cam.squeeze(1).sum(dim=[1, 2])
        best_idx = cam_strength.argmax().item()

        best_cam = cam[best_idx, 0].cpu().numpy()
        if best_cam.max() > 0:
            best_cam = (best_cam - best_cam.min()) / (best_cam.max() - best_cam.min())

        probs = torch.softmax(output, dim=1)
        return best_cam, probs, best_idx

# %% [markdown]
# # Cell 4: Load Model

# %%
print("Loading V9 model...")
model = MRNetV9()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model = model.to(device)
model.eval()
gradcam = GradCAM(model, model.features[-1])
print("  Done")

# %% [markdown]
# # Cell 5: Select 20 Samples (10 Normal + 10 Tear)

# %%
meta = pd.read_csv(Path(DATA_DIR) / 'metadata.csv')
meta['label_binary'] = meta['label'].astype(int)

normals = meta[meta['label_binary'] == 0].sample(n=10, random_state=SEED)
tears = meta[meta['label_binary'] == 1].sample(n=10, random_state=SEED)
samples = pd.concat([normals, tears]).reset_index(drop=True)

print(f"Selected {len(samples)} samples:")
print(f"  Normal: {len(normals)}, Tear: {len(tears)}")

# %% [markdown]
# # Cell 6: Run GradCAM on All 20 Samples

# %%
results = []

for i, row in samples.iterrows():
    filepath = Path(DATA_DIR) / row['filename']
    volume = np.load(filepath)['data']
    slices = volume.astype(np.float32) / 255.0

    if slices.shape[0] > MAX_SLICES:
        offset = (slices.shape[0] - MAX_SLICES) // 2
        slices = slices[offset:offset + MAX_SLICES]

    slices_3ch = np.stack((slices,)*3, axis=1)
    tensor = torch.FloatTensor(slices_3ch).unsqueeze(0).to(device)

    heatmap, probs, best_slice_idx = gradcam.generate(tensor, target_class=1)

    prob_tear = probs[0][1].item()
    pred = 'TEAR' if prob_tear >= 0.5 else 'NORMAL'
    true_label = 'TEAR' if row['label_binary'] == 1 else 'NORMAL'
    correct = pred == true_label

    # Get the actual slice image for display
    display_slice = slices[best_slice_idx]

    results.append({
        'patient': row['patient_id'],
        'true': true_label,
        'pred': pred,
        'prob_tear': prob_tear,
        'correct': correct,
        'slice_img': display_slice,
        'heatmap': heatmap,
        'best_slice': best_slice_idx,
        'total_slices': slices.shape[0]
    })
    status = '\u2705' if correct else '\u274c'
    print(f"  [{i+1:2d}/20] {row['patient_id']}: True={true_label:6s} Pred={pred:6s} P(Tear)={prob_tear:.3f} Slice={best_slice_idx}/{slices.shape[0]} {status}")

correct_count = sum(r['correct'] for r in results)
print(f"\nAccuracy: {correct_count}/{len(results)} ({100*correct_count/len(results):.0f}%)")

# %% [markdown]
# # Cell 7: GradCAM Grid — All 20 Samples

# %%
fig, axes = plt.subplots(5, 8, figsize=(32, 20))

for i, r in enumerate(results):
    row_idx = i // 4
    col_idx = (i % 4) * 2

    # Original slice
    ax1 = axes[row_idx, col_idx]
    ax1.imshow(r['slice_img'], cmap='gray')
    color = 'green' if r['correct'] else 'red'
    ax1.set_title(f"{r['patient']}\nTrue: {r['true']}", fontsize=9, color=color, fontweight='bold')
    ax1.axis('off')

    # GradCAM overlay
    ax2 = axes[row_idx, col_idx + 1]
    heatmap_resized = cv2.resize(r['heatmap'], (r['slice_img'].shape[1], r['slice_img'].shape[0]))
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
    original_rgb = np.stack([r['slice_img']]*3, axis=-1)
    overlay = 0.6 * original_rgb + 0.4 * heatmap_colored
    overlay = np.clip(overlay, 0, 1)
    ax2.imshow(overlay)
    ax2.set_title(f"Pred: {r['pred']} ({r['prob_tear']:.2f})\nSlice {r['best_slice']}/{r['total_slices']}", fontsize=9, color=color, fontweight='bold')
    ax2.axis('off')

plt.suptitle(f"V9 GradCAM — MRNet Dataset ({correct_count}/{len(results)} correct)", fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(r'c:\ML projects\ACL tears\V9\gradcam_grid_20.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved to V9/gradcam_grid_20.png")

# %% [markdown]
# # Cell 8: Summary Table

# %%
summary = pd.DataFrame([{
    'Patient': r['patient'],
    'True Label': r['true'],
    'Prediction': r['pred'],
    'P(Tear)': f"{r['prob_tear']:.4f}",
    'Correct': '\u2705' if r['correct'] else '\u274c',
    'Best Slice': f"{r['best_slice']}/{r['total_slices']}"
} for r in results])

print(summary.to_string(index=False))
