# ============================================================
# V9 Model Evaluation on Priyank Saxena Dataset (Local Laptop)
# ============================================================
# Uses best_acl_model_v9.pth trained on MRNet sagittal data
# Evaluates on Priyank Saxena external dataset as validation

# %% [markdown]
# # Cell 1: Configuration

# %%
import os
MODEL_PATH = r'c:\ML projects\ACL tears\V9\best_acl_model_v9.pth'
DATA_DIR = r'c:\ML projects\ACL tears\DATASET\combined'
MAX_SLICES = 40
OPTIMAL_THRESHOLD = 0.706  # From V9 training validation
# Priyank ACL is in slices 34-40 (of 50). Extract a region around it.
PRIYANK_SLICE_START = 25
PRIYANK_SLICE_END = 50    # slices 25-49 (25 slices, centered on ACL)

# %% [markdown]
# # Cell 2: Imports

# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# %% [markdown]
# # Cell 3: Load Priyank Metadata

# %%
metadata = pd.read_csv(Path(DATA_DIR) / 'metadata.csv')
priyank_df = metadata[metadata['source'] == 'Priyank_Saxena'].copy()

# Map 3-class to binary: NORMAL(0) -> 0, PARTIAL(1) + COMPLETE(2) -> 1 (Tear)
priyank_df['label_binary'] = (priyank_df['label'] > 0).astype(int)
priyank_df['label_name_binary'] = priyank_df['label_binary'].map({0: 'Normal', 1: 'Tear'})

print(f"Priyank Saxena Dataset: {len(priyank_df)} patients")
print(priyank_df['label_name_binary'].value_counts())
print(f"\nOriginal 3-class distribution:")
print(priyank_df['label_name'].value_counts())
print(f"\nSlice count: min={priyank_df['num_slices'].min()}, max={priyank_df['num_slices'].max()}, mean={priyank_df['num_slices'].mean():.1f}")

# %% [markdown]
# # Cell 4: Dataset (matching V9 preprocessing exactly)

# %%
class MRNetVolumeDataset(Dataset):
    def __init__(self, df, data_dir, max_slices=MAX_SLICES):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.max_slices = max_slices
        self.valid_indices = []
        for idx, row in self.df.iterrows():
            if (self.data_dir / row['filename']).exists():
                self.valid_indices.append(idx)
        print(f"  {len(self.valid_indices)} valid patients (of {len(self.df)})")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        patient_idx = self.valid_indices[idx]
        row = self.df.iloc[patient_idx]
        volume = np.load(self.data_dir / row['filename'])['data']
        slices = volume.astype(np.float32) / 255.0
        actual_slices = slices.shape[0]

        # For Priyank data (50 slices), extract ACL region only
        if actual_slices >= 50:
            slices = slices[PRIYANK_SLICE_START:PRIYANK_SLICE_END]
            actual_slices = slices.shape[0]

        if actual_slices > self.max_slices:
            offset = (actual_slices - self.max_slices) // 2
            slices = slices[offset:offset + self.max_slices]

        # Resize to 256x256 if needed (V9 trained on MRNet 256x256)
        h, w = slices.shape[1], slices.shape[2]
        if h != 256 or w != 256:
            import cv2
            slices = np.stack([cv2.resize(s, (256, 256), interpolation=cv2.INTER_LINEAR) for s in slices])

        # Stack to 3 channels (matching V9 training)
        slices_3ch = np.stack((slices,)*3, axis=1)
        slices_tensor = torch.FloatTensor(slices_3ch)

        label = int(row['label_binary'])
        patient_id = row['patient_id']
        original_label = row['label_name']

        return slices_tensor, label, patient_id, original_label

# %% [markdown]
# # Cell 5: Model (must match V9 architecture exactly)

# %%
class MRNetV9(nn.Module):
    """Must match the EXACT architecture used during training."""
    def __init__(self, dropout=0.3):
        super().__init__()
        backbone = models.efficientnet_b0(weights=None)  # No pretrained needed for eval
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

# %% [markdown]
# # Cell 6: Load Model & Create Dataset

# %%
print("Loading V9 model...")
model = MRNetV9()
state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
print(f"  Loaded from: {MODEL_PATH}")

print("\nCreating Priyank dataset...")
dataset = MRNetVolumeDataset(priyank_df, DATA_DIR)
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
print(f"  {len(dataset)} patients to evaluate")

# %% [markdown]
# # Cell 7: Run Inference

# %%
y_trues = []
y_probs = []
patient_ids = []
original_labels = []

print("Running inference...")
with torch.no_grad():
    for volumes, labels, p_ids, orig_labels in tqdm(loader, desc='Evaluating'):
        volumes = volumes.to(device)
        logits = model(volumes.float())
        probs = torch.softmax(logits, dim=1)
        y_trues.append(labels[0].item())
        y_probs.append(probs[0][1].item())  # P(Tear)
        patient_ids.append(p_ids[0])
        original_labels.append(orig_labels[0])

print(f"Done. {len(y_trues)} patients evaluated.")

# %% [markdown]
# # Cell 8: Results

# %%
label_names = ['Normal', 'Tear']
auc = roc_auc_score(y_trues, y_probs)

# Find optimal threshold on this dataset too
fpr, tpr, thresholds = roc_curve(y_trues, y_probs)
youden_j = tpr - fpr
best_idx = np.argmax(youden_j)
priyank_threshold = thresholds[best_idx]

# Predictions at different thresholds
preds_default = [1 if p >= 0.5 else 0 for p in y_probs]
preds_v9_threshold = [1 if p >= OPTIMAL_THRESHOLD else 0 for p in y_probs]
preds_priyank_optimal = [1 if p >= priyank_threshold else 0 for p in y_probs]

print('=' * 70)
print('V9 Model Evaluation on Priyank Saxena Dataset')
print('=' * 70)
print(f'AUC: {auc:.4f}')
print(f'Optimal threshold (Priyank): {priyank_threshold:.4f} (Youden J = {youden_j[best_idx]:.4f})')

print(f'\n--- Default threshold (0.5) ---')
print(classification_report(y_trues, preds_default, target_names=label_names, digits=3))

print(f'--- V9 training threshold ({OPTIMAL_THRESHOLD}) ---')
print(classification_report(y_trues, preds_v9_threshold, target_names=label_names, digits=3))

print(f'--- Priyank optimal threshold ({priyank_threshold:.3f}) ---')
print(classification_report(y_trues, preds_priyank_optimal, target_names=label_names, digits=3))

# %% [markdown]
# # Cell 9: Confusion Matrix

# %%
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

for ax, preds, title in zip(axes,
    [preds_default, preds_v9_threshold, preds_priyank_optimal],
    [f'Default (0.5)', f'V9 Threshold ({OPTIMAL_THRESHOLD})', f'Priyank Optimal ({priyank_threshold:.3f})']):
    cm = confusion_matrix(y_trues, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(f'CM - {title}')

plt.suptitle(f'V9 on Priyank Saxena (AUC={auc:.4f})', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(r'c:\ML projects\ACL tears\V9\priyank_evaluation_v9.png', dpi=150)
plt.show()

# %% [markdown]
# # Cell 10: Per-Patient Details

# %%
results_df = pd.DataFrame({
    'patient_id': patient_ids,
    'original_label': original_labels,
    'true_binary': y_trues,
    'prob_tear': y_probs,
    'pred_default': preds_default,
    'pred_v9': preds_v9_threshold,
    'pred_optimal': preds_priyank_optimal
})

# Show misclassified cases at optimal threshold
misclassified = results_df[results_df['true_binary'] != results_df['pred_optimal']]
print(f"\nMisclassified at optimal threshold: {len(misclassified)} / {len(results_df)} ({100*len(misclassified)/len(results_df):.1f}%)")
print("\nMisclassified cases:")
print(misclassified.to_string(index=False))

# Save full results
results_df.to_csv(r'c:\ML projects\ACL tears\V9\priyank_predictions_v9.csv', index=False)
print(f"\nFull results saved to priyank_predictions_v9.csv")

# %% [markdown]
# # Cell 11: ROC Curve

# %%
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
plt.scatter(fpr[best_idx], tpr[best_idx], color='red', s=100, zorder=5,
            label=f'Optimal ({priyank_threshold:.3f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - V9 on Priyank Saxena')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(r'c:\ML projects\ACL tears\V9\roc_curve_priyank_v9.png', dpi=150)
plt.show()
