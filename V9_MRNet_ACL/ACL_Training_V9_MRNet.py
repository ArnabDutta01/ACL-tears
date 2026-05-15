# ============================================================
# ACL Tear Detection — V9 MRNet (Faithful Reproduction)
# ============================================================
# V8.4 problems: AUC stuck at ~0.73, massive overfitting
#
# V9 changes (matching original Stanford MRNet):
# | Setting         | V8.4                          | V9 (Original MRNet)              |
# |-----------------|-------------------------------|----------------------------------|
# | Backbone        | EfficientNet-B0 (3 blocks frozen) | AlexNet (fully trainable)    |
# | Pooling         | Attention-weighted            | Simple max across slices         |
# | Classifier      | Bottleneck 1280→256→LN→2      | Linear(256, 2)                   |
# | Loss            | Focal + label smoothing       | BCEWithLogitsLoss (class-weighted)|
# | Optimizer       | AdamW (diff LR, wd=0.05)      | Adam (lr=1e-5, wd=0.1)          |
# | Scheduler       | Warmup + Cosine               | ReduceLROnPlateau (p=3, f=0.3)  |
# | Grad Accum      | 8 steps                       | None (batch=1, instant update)  |
# | Normalization   | /255 + ImageNet normalize     | /255 only (no ImageNet stats)   |
# | Train split     | 70%                           | 85%                              |

# %% [markdown]
# # Cell 1: Mount Drive (Colab)

# %%
from google.colab import drive
drive.mount("/content/drive")

# %% [markdown]
# # Cell 2: Configuration

# %%
MRNET_DATA_DIR = '/content/drive/MyDrive/dataset/mrnet_sagittal'
BATCH_SIZE = 1
NUM_EPOCHS = 50
RANDOM_SEED = 42
MAX_SLICES = 40
LR = 5e-5
WEIGHT_DECAY = 0.01
PATIENCE = 10
DROPOUT = 0.5
FREEZE_BLOCKS = 4

# %% [markdown]
# # Cell 3: Imports

# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# %% [markdown]
# # Cell 4: Load Metadata

# %%
mrnet_path = Path(MRNET_DATA_DIR)
mrnet_df = pd.read_csv(mrnet_path / 'metadata.csv')
mrnet_df['label_binary'] = mrnet_df['label'].astype(int)
mrnet_df['label_name_binary'] = mrnet_df['label_binary'].map({0: 'Normal', 1: 'Tear'})

print(f"MRNet Dataset: {len(mrnet_df)} patients")
print(mrnet_df['label_name_binary'].value_counts())
print(f"\nSlice count: min={mrnet_df['num_slices'].min()}, max={mrnet_df['num_slices'].max()}, mean={mrnet_df['num_slices'].mean():.1f}")

# %% [markdown]
# # Cell 5: Dataset — No ImageNet normalization, simple augmentation

# %%
class MRNetVolumeDataset(Dataset):
    def __init__(self, df, data_dir, max_slices=MAX_SLICES, augment=False):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.max_slices = max_slices
        self.augment = augment
        # NO ImageNet normalization — matching original MRNet
        self.valid_indices = []
        for idx, row in self.df.iterrows():
            if (self.data_dir / row['filename']).exists():
                self.valid_indices.append(idx)
        print(f"  {len(self.valid_indices)} valid patients")

    def __len__(self):
        return len(self.valid_indices)

    def _augment_volume(self, slices):
        import cv2
        # Horizontal flip (matching original RandomFlip)
        if np.random.random() < 0.5:
            slices = slices[:, :, ::-1].copy()
        # Rotation up to 25° (matching original RandomRotate(25))
        if np.random.random() < 0.5:
            angle = np.random.uniform(-25, 25)
            h, w = slices.shape[1], slices.shape[2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            slices = np.stack([cv2.warpAffine(s, M, (w, h), borderValue=0) for s in slices])
        # Translation up to 11% (matching original RandomTranslate([0.11, 0.11]))
        if np.random.random() < 0.5:
            h, w = slices.shape[1], slices.shape[2]
            tx = np.random.uniform(-0.11, 0.11) * w
            ty = np.random.uniform(-0.11, 0.11) * h
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            slices = np.stack([cv2.warpAffine(s, M, (w, h), borderValue=0) for s in slices])
        return slices

    def __getitem__(self, idx):
        patient_idx = self.valid_indices[idx]
        row = self.df.iloc[patient_idx]
        volume = np.load(self.data_dir / row['filename'])['data']
        slices = volume.astype(np.float32) / 255.0
        actual_slices = slices.shape[0]

        # No padding needed — we'll use variable-length max pooling
        # But we still cap at max_slices for memory
        if actual_slices > self.max_slices:
            offset = (actual_slices - self.max_slices) // 2
            slices = slices[offset:offset + self.max_slices]
            actual_slices = self.max_slices

        if self.augment:
            slices = self._augment_volume(slices)

        # Stack to 3 channels (grayscale → RGB for pretrained backbone)
        slices_3ch = np.stack((slices,)*3, axis=1)
        slices_tensor = torch.FloatTensor(slices_3ch)
        # NO normalization — just raw [0,1] values

        label = int(row['label_binary'])
        return slices_tensor, label, patient_idx

    def get_labels(self):
        return [int(self.df.iloc[i]['label_binary']) for i in self.valid_indices]

# %% [markdown]
# # Cell 6: Model — EfficientNet-B0 + max pool + Linear (blocks 0-3 frozen)

# %%
class MRNetV9(nn.Module):
    """MRNet-style architecture with EfficientNet-B0.
    Blocks 0-(FREEZE_BLOCKS-1) frozen to prevent overfitting.
    Simple: features -> avgpool -> max across slices -> Dropout -> Linear(1280, 2)
    """
    def __init__(self, dropout=DROPOUT, freeze_blocks=FREEZE_BLOCKS):
        super().__init__()
        backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.features = backbone.features
        self.pool = backbone.avgpool
        self.drop = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(1280, 2)

        # Freeze early blocks to prevent memorization
        for i in range(min(freeze_blocks, len(self.features))):
            for param in self.features[i].parameters():
                param.requires_grad = False

        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = frozen + trainable
        print(f"  Backbone: EfficientNet-B0, blocks 0-{freeze_blocks-1} FROZEN")
        print(f"  Pooling: max across slices (no attention)")
        print(f"  Classifier: Dropout({dropout}) -> Linear(1280, 2)")
        print(f"  Params: {trainable:,} trainable / {total:,} total ({100*trainable/total:.1f}%)")

    def forward(self, x):
        x = x.squeeze(0)  # (S, 3, H, W)
        features = self.features(x)                 # (S, 1280, 7, 7)
        pooled = self.pool(features)                # (S, 1280, 1, 1)
        pooled = pooled.flatten(1)                  # (S, 1280)
        volume_feat = torch.max(pooled, 0, keepdim=True)[0]  # (1, 1280)
        volume_feat = self.drop(volume_feat)
        output = self.classifier(volume_feat)       # (1, 2)
        return output

# %% [markdown]
# # Cell 7: Train/Val Split (85/15 — maximize training data)

# %%
train_df, val_df = train_test_split(
    mrnet_df, test_size=0.15, stratify=mrnet_df['label_binary'], random_state=RANDOM_SEED
)

print("MRNet Dataset Split:")
for name, df in [('Train', train_df), ('Val', val_df)]:
    dist = df['label_name_binary'].value_counts().to_dict()
    print(f"  {name}: {len(df)} patients — {dist}")

# %% [markdown]
# # Cell 8: Create Datasets & DataLoaders

# %%
print("Creating datasets...")
print(f"  MAX_SLICES={MAX_SLICES}")

print("\nTrain (augmented):")
train_dataset = MRNetVolumeDataset(train_df, MRNET_DATA_DIR, augment=True)
print("Val:")
val_dataset = MRNetVolumeDataset(val_df, MRNET_DATA_DIR, augment=False)

# Compute class weights for CrossEntropyLoss
labels = train_dataset.get_labels()
pos = sum(labels)
neg = len(labels) - pos
class_weights = torch.FloatTensor([1.0, neg / pos])
print(f"\nClass weights: Normal={class_weights[0]:.2f}, Tear={class_weights[1]:.2f} (neg/pos={neg}/{pos})")

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

print(f"\nTrain: {len(train_dataset)} patients, {len(train_loader)} batches")
print(f"Val:   {len(val_dataset)} patients")

# %% [markdown]
# # Cell 9: Model, Loss, Optimizer

# %%
print("Creating V9 model...")
model = MRNetV9()
model = model.to(device)

# CrossEntropyLoss with class weights — properly upweights minority (Tear) class
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

# Single Adam optimizer — matching original (lr=1e-5, wd=0.1)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# ReduceLROnPlateau (patience=5 for LR, PATIENCE=10 for early stop)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, factor=0.3, threshold=1e-4
)

print(f"\nV9 Settings:")
print(f"  Loss:        BCEWithLogitsLoss (weights=[{class_weights[0]:.2f}, {class_weights[1]:.2f}])")
print(f"  Optimizer:   Adam (lr={LR}, wd={WEIGHT_DECAY})")
print(f"  Scheduler:   ReduceLROnPlateau (patience=3, factor=0.3)")
print(f"  Batch size:  1 (no gradient accumulation)")
print(f"  Early stop:  patience={PATIENCE} on val_AUC")

# %% [markdown]
# # Cell 10: Training & Validation Functions

# %%
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    y_trues = []
    y_preds = []

    for volumes, labels, _ in tqdm(loader, desc='Training', leave=False):
        optimizer.zero_grad()

        volumes = volumes.to(device)
        labels = labels.to(device)  # (1,) integer class index

        logits = model(volumes.float())  # (1, 2)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        probs = torch.softmax(logits, dim=1)
        y_trues.append(labels[0].item())
        y_preds.append(probs[0][1].item())  # P(Tear)

    avg_loss = np.mean(losses)
    try:
        auc = roc_auc_score(y_trues, y_preds)
    except:
        auc = 0.5

    preds_binary = [1 if p > 0.5 else 0 for p in y_preds]
    acc = 100.0 * sum(p == t for p, t in zip(preds_binary, y_trues)) / len(y_trues)
    return avg_loss, acc, auc


def validate(model, loader, criterion, device):
    model.eval()
    losses = []
    y_trues = []
    y_probs = []

    with torch.no_grad():
        for volumes, labels, _ in tqdm(loader, desc='Validating', leave=False):
            volumes = volumes.to(device)
            labels = labels.to(device)

            logits = model(volumes.float())
            loss = criterion(logits, labels)

            losses.append(loss.item())
            probs = torch.softmax(logits, dim=1)
            y_trues.append(labels[0].item())
            y_probs.append(probs[0][1].item())  # P(Tear)

    avg_loss = np.mean(losses)
    try:
        auc = roc_auc_score(y_trues, y_probs)
    except:
        auc = 0.5

    preds_binary = [1 if p > 0.5 else 0 for p in y_probs]
    acc = 100.0 * sum(p == t for p, t in zip(preds_binary, y_trues)) / len(y_trues)
    return avg_loss, acc, auc, y_probs, y_trues

# %% [markdown]
# # Cell 11: Training Loop

# %%
history = {'train_loss': [], 'train_acc': [], 'train_auc': [],
           'val_loss': [], 'val_acc': [], 'val_auc': [], 'lr': []}
best_val_auc = 0.0
patience_counter = 0
best_val_loss = float('inf')
iteration_no_improve_loss = 0

SAVE_PATH = '/content/drive/MyDrive/dataset/best_acl_model_v9.pth'

print(f"Training for up to {NUM_EPOCHS} epochs (patience={PATIENCE})...\n")

for epoch in range(NUM_EPOCHS):
    current_lr = optimizer.param_groups[0]['lr']

    train_loss, train_acc, train_auc = train_epoch(
        model, train_loader, criterion, optimizer, device
    )
    val_loss, val_acc, val_auc, val_preds, val_labels = validate(
        model, val_loader, criterion, device
    )

    # ReduceLROnPlateau steps on val_loss — matching original
    scheduler.step(val_loss)

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['train_auc'].append(train_auc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_auc'].append(val_auc)
    history['lr'].append(current_lr)

    gap = train_acc - val_acc
    gap_warn = ' OVERFIT' if gap > 10 else ''

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}  (lr={current_lr:.1e})")
    print(f"  Train: loss={train_loss:.4f}  acc={train_acc:.2f}%  AUC={train_auc:.4f}")
    print(f"  Val:   loss={val_loss:.4f}  acc={val_acc:.2f}%  AUC={val_auc:.4f}  (gap={gap:.1f}%){gap_warn}")

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  -> Saved best (AUC={val_auc:.4f}, acc={val_acc:.2f}%, loss={val_loss:.4f})")
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"  -> No improvement ({patience_counter}/{PATIENCE})")
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

print(f"\nBest val AUC: {best_val_auc:.4f}")

# %% [markdown]
# # Cell 12: Plot Training History

# %%
fig, axes = plt.subplots(1, 5, figsize=(30, 5))

axes[0].plot(history['train_loss'], label='Train')
axes[0].plot(history['val_loss'], label='Val')
axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True)

axes[1].plot(history['train_acc'], label='Train')
axes[1].plot(history['val_acc'], label='Val')
axes[1].set_title('Accuracy (%)'); axes[1].legend(); axes[1].grid(True)

axes[2].plot(history['train_auc'], label='Train')
axes[2].plot(history['val_auc'], label='Val')
axes[2].set_title('AUC'); axes[2].legend(); axes[2].grid(True)

gaps = [t - v for t, v in zip(history['train_acc'], history['val_acc'])]
axes[3].plot(gaps, color='red')
axes[3].axhline(y=10, color='orange', linestyle='--', label='10% threshold')
axes[3].set_title('Overfit Gap'); axes[3].legend(); axes[3].grid(True)

axes[4].plot(history['lr'], color='purple')
axes[4].set_title('Learning Rate'); axes[4].grid(True)
axes[4].set_yscale('log')

for ax in axes: ax.set_xlabel('Epoch')
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/dataset/training_history_v9.png', dpi=150)
plt.show()

# %% [markdown]
# # Cell 13: Evaluate Best Model

# %%
model.load_state_dict(torch.load(SAVE_PATH, weights_only=True))
val_loss, val_acc, val_auc, val_probs, val_labels = validate(
    model, val_loader, criterion, device
)

label_names = ['Normal', 'Tear']

# --- Find optimal threshold (maximize Youden's J = sensitivity + specificity - 1) ---
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(val_labels, val_probs)
youden_j = tpr - fpr
best_idx = np.argmax(youden_j)
best_threshold = thresholds[best_idx]
print(f'Optimal threshold: {best_threshold:.4f} (Youden J = {youden_j[best_idx]:.4f})')
print(f'At this threshold: Sensitivity={tpr[best_idx]:.3f}, Specificity={1-fpr[best_idx]:.3f}')

# Apply optimal threshold
val_preds_optimal = [1 if p >= best_threshold else 0 for p in val_probs]
val_preds_default = [1 if p >= 0.5 else 0 for p in val_probs]

print('\n' + '=' * 60)
print('RESULTS — V9 (Faithful MRNet)')
print('=' * 60)
print(f'AUC: {val_auc:.4f}')

print(f'\n--- Default threshold (0.5) ---')
print(classification_report(val_labels, val_preds_default, target_names=label_names, digits=3))

print(f'--- Optimal threshold ({best_threshold:.3f}) ---')
print(classification_report(val_labels, val_preds_optimal, target_names=label_names, digits=3))

cm = confusion_matrix(val_labels, val_preds_optimal)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_names, yticklabels=label_names)
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.title(f'Confusion Matrix — V9 (threshold={best_threshold:.3f})')
plt.savefig('/content/drive/MyDrive/dataset/confusion_matrix_v9.png', dpi=150)
plt.show()
