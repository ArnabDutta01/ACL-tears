# ============================================================
# ACL Tear Detection — V11 Multi-Task MRNet
# ============================================================
# Same architecture as V10 (which got ACL AUC=0.907) plus:
#   1. Composite model selection (saves on weighted mean AUC)
#   2. ReduceLROnPlateau on composite AUC
# ============================================================

# %% [markdown]
# # Cell 1: Mount Drive (Colab)

# %%
from google.colab import drive
drive.mount("/content/drive")

# %% [markdown]
# # Cell 2: Configuration

# %%
DATA_DIR = '/content/drive/MyDrive/dataset/mrnet_all'
BATCH_SIZE = 1
NUM_EPOCHS = 50
RANDOM_SEED = 42
MAX_SLICES = 40
LR = 1e-4
WEIGHT_DECAY = 0.01
PATIENCE = 10
DROPOUT = 0.3

# Loss weights for each task
TASK_WEIGHT_ACL = 1.0
TASK_WEIGHT_MENISCUS = 1.0
TASK_WEIGHT_ABNORMAL = 0.5

# Model selection weights (for composite AUC)
SEL_WEIGHT_ACL = 0.5
SEL_WEIGHT_MEN = 0.3
SEL_WEIGHT_ABN = 0.2

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
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if device.type == 'cuda':
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')

# %% [markdown]
# # Cell 4: Load Metadata + Split

# %%
data_path = Path(DATA_DIR)
metadata = pd.read_csv(data_path / 'metadata.csv')

print(f"Total patients: {len(metadata)}")
print(f"\nLabel distribution:")
print(f"  ACL tear:      {metadata.label_acl.sum():4d} / {len(metadata)} ({100*metadata.label_acl.mean():.1f}%)")
print(f"  Meniscus tear: {metadata.label_meniscus.sum():4d} / {len(metadata)} ({100*metadata.label_meniscus.mean():.1f}%)")
print(f"  Abnormal:      {metadata.label_abnormal.sum():4d} / {len(metadata)} ({100*metadata.label_abnormal.mean():.1f}%)")
print(f"  Normal (all 0): {((metadata.label_abnormal==0)).sum():4d} / {len(metadata)}")

# Stratified split on ACL label (primary task)
train_df, val_df = train_test_split(
    metadata, test_size=0.15, random_state=RANDOM_SEED,
    stratify=metadata['label_acl']
)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print(f"\nTrain: {len(train_df)} patients")
print(f"  ACL: {train_df.label_acl.sum()} tear / {(train_df.label_acl==0).sum()} normal")
print(f"  Meniscus: {train_df.label_meniscus.sum()} tear / {(train_df.label_meniscus==0).sum()} normal")
print(f"  Abnormal: {train_df.label_abnormal.sum()} yes / {(train_df.label_abnormal==0).sum()} no")
print(f"\nVal: {len(val_df)} patients")
print(f"  ACL: {val_df.label_acl.sum()} tear / {(val_df.label_acl==0).sum()} normal")

# %% [markdown]
# # Cell 5: Dataset — Strong Augmentation + Multi-Task Labels

# %%
class MultiTaskDataset(Dataset):
    """Loads sagittal view from mrnet_all .npz files with 3 labels."""
    def __init__(self, df, data_dir, max_slices=MAX_SLICES, augment=False):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.max_slices = max_slices
        self.augment = augment
        self.valid_indices = []

        for idx in range(len(self.df)):
            fpath = self.data_dir / self.df.iloc[idx]['filename']
            if fpath.exists():
                self.valid_indices.append(idx)

        print(f"  {len(self.valid_indices)} valid patients (of {len(self.df)})")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        patient_idx = self.valid_indices[idx]
        row = self.df.iloc[patient_idx]

        data = np.load(self.data_dir / row['filename'])
        volume = data['sagittal']  # (S, 256, 256), uint8
        slices = volume.astype(np.float32) / 255.0
        actual_slices = slices.shape[0]

        # Center crop if too many slices
        if actual_slices > self.max_slices:
            offset = (actual_slices - self.max_slices) // 2
            slices = slices[offset:offset + self.max_slices]

        # Simple augmentation: random horizontal flip
        if self.augment and np.random.random() > 0.5:
            slices = slices[:, :, ::-1].copy()

        # Stack to 3 channels
        slices_3ch = np.stack((slices,)*3, axis=1)  # (S, 3, 256, 256)
        tensor = torch.FloatTensor(slices_3ch)

        # Multi-task labels
        labels = torch.LongTensor([
            int(row['label_acl']),
            int(row['label_meniscus']),
            int(row['label_abnormal'])
        ])

        return tensor, labels

    def get_label_counts(self):
        """Get per-task label counts for class weight computation."""
        acl = [int(self.df.iloc[i]['label_acl']) for i in self.valid_indices]
        men = [int(self.df.iloc[i]['label_meniscus']) for i in self.valid_indices]
        abn = [int(self.df.iloc[i]['label_abnormal']) for i in self.valid_indices]
        return {'acl': acl, 'meniscus': men, 'abnormal': abn}

# Create datasets
print("Creating datasets...")
print("Train:")
train_dataset = MultiTaskDataset(train_df, DATA_DIR, augment=True)
print("Val:")
val_dataset = MultiTaskDataset(val_df, DATA_DIR, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=2, pin_memory=True)

# %% [markdown]
# # Cell 6: Compute Class Weights

# %%
label_counts = train_dataset.get_label_counts()

def compute_weight(labels):
    n = len(labels)
    n_pos = sum(labels)
    n_neg = n - n_pos
    w_neg = n / (2 * n_neg) if n_neg > 0 else 1.0
    w_pos = n / (2 * n_pos) if n_pos > 0 else 1.0
    return torch.FloatTensor([w_neg, w_pos]).to(device)

weight_acl = compute_weight(label_counts['acl'])
weight_meniscus = compute_weight(label_counts['meniscus'])
weight_abnormal = compute_weight(label_counts['abnormal'])

print(f"Class weights:")
print(f"  ACL:      Normal={weight_acl[0]:.3f}, Tear={weight_acl[1]:.3f}")
print(f"  Meniscus: Normal={weight_meniscus[0]:.3f}, Tear={weight_meniscus[1]:.3f}")
print(f"  Abnormal: Normal={weight_abnormal[0]:.3f}, Abnormal={weight_abnormal[1]:.3f}")

# %% [markdown]
# # Cell 7: Model — V10 Architecture + Dropout

# %%
class MRNetV11(nn.Module):
    """Multi-task MRNet: shared EfficientNet-B0 backbone, 3 classifier heads.
    Same proven architecture as V10. Max-pool across slices.
    """
    def __init__(self, dropout=DROPOUT):
        super().__init__()
        backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.features = backbone.features
        self.pool = backbone.avgpool
        self.drop = nn.Dropout(p=dropout)

        # 3 task-specific heads (simple linear, same as V10)
        self.head_acl = nn.Linear(1280, 2)
        self.head_meniscus = nn.Linear(1280, 2)
        self.head_abnormal = nn.Linear(1280, 2)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Backbone: EfficientNet-B0 (all trainable)")
        print(f"  Pooling: max across slices")
        print(f"  Heads: ACL(1280->2), Meniscus(1280->2), Abnormal(1280->2)")
        print(f"  Dropout: {dropout}")
        print(f"  Params: {trainable:,} trainable / {total:,} total")

    def forward(self, x):
        # x: (1, S, 3, H, W)
        x = x.squeeze(0)  # (S, 3, H, W)
        features = self.features(x)      # (S, 1280, 8, 8)
        pooled = self.pool(features)     # (S, 1280, 1, 1)
        pooled = pooled.flatten(1)       # (S, 1280)

        # Max pool across slices
        volume_feat = torch.max(pooled, 0, keepdim=True)[0]  # (1, 1280)
        volume_feat = self.drop(volume_feat)

        # 3 task predictions
        out_acl = self.head_acl(volume_feat)
        out_meniscus = self.head_meniscus(volume_feat)
        out_abnormal = self.head_abnormal(volume_feat)

        return out_acl, out_meniscus, out_abnormal

print("Creating model...")
model = MRNetV11().to(device)

# %% [markdown]
# # Cell 8: Loss + Optimizer + Scheduler

# %%
criterion_acl = nn.CrossEntropyLoss(weight=weight_acl)
criterion_meniscus = nn.CrossEntropyLoss(weight=weight_meniscus)
criterion_abnormal = nn.CrossEntropyLoss(weight=weight_abnormal)

optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.3, patience=5, min_lr=1e-7
)

print(f"Optimizer: Adam (lr={LR}, weight_decay={WEIGHT_DECAY})")
print(f"Scheduler: ReduceLROnPlateau (patience=5, factor=0.3, on composite AUC)")
print(f"Task weights: ACL={TASK_WEIGHT_ACL}, Men={TASK_WEIGHT_MENISCUS}, Abn={TASK_WEIGHT_ABNORMAL}")
print(f"Selection weights: ACL={SEL_WEIGHT_ACL}, Men={SEL_WEIGHT_MEN}, Abn={SEL_WEIGHT_ABN}")

# %% [markdown]
# # Cell 9: Train + Validate Functions

# %%
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    all_labels = {'acl': [], 'meniscus': [], 'abnormal': []}
    all_probs = {'acl': [], 'meniscus': [], 'abnormal': []}

    for volumes, labels in tqdm(loader, desc='Training', leave=False):
        volumes = volumes.to(device)
        lab_acl = labels[:, 0].to(device)
        lab_men = labels[:, 1].to(device)
        lab_abn = labels[:, 2].to(device)

        optimizer.zero_grad()
        out_acl, out_men, out_abn = model(volumes.float())

        loss_acl = criterion_acl(out_acl, lab_acl)
        loss_men = criterion_meniscus(out_men, lab_men)
        loss_abn = criterion_abnormal(out_abn, lab_abn)

        loss = (TASK_WEIGHT_ACL * loss_acl +
                TASK_WEIGHT_MENISCUS * loss_men +
                TASK_WEIGHT_ABNORMAL * loss_abn)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Collect predictions
        probs_acl = torch.softmax(out_acl, dim=1)[:, 1].detach().cpu().numpy()
        probs_men = torch.softmax(out_men, dim=1)[:, 1].detach().cpu().numpy()
        probs_abn = torch.softmax(out_abn, dim=1)[:, 1].detach().cpu().numpy()

        all_labels['acl'].extend(lab_acl.cpu().numpy())
        all_labels['meniscus'].extend(lab_men.cpu().numpy())
        all_labels['abnormal'].extend(lab_abn.cpu().numpy())
        all_probs['acl'].extend(probs_acl)
        all_probs['meniscus'].extend(probs_men)
        all_probs['abnormal'].extend(probs_abn)

    avg_loss = total_loss / len(loader)
    aucs = {}
    for task in ['acl', 'meniscus', 'abnormal']:
        try:
            aucs[task] = roc_auc_score(all_labels[task], all_probs[task])
        except ValueError:
            aucs[task] = 0.5

    return avg_loss, aucs


def validate(model, loader, device):
    model.eval()
    total_loss = 0
    all_labels = {'acl': [], 'meniscus': [], 'abnormal': []}
    all_probs = {'acl': [], 'meniscus': [], 'abnormal': []}

    with torch.no_grad():
        for volumes, labels in tqdm(loader, desc='Validating', leave=False):
            volumes = volumes.to(device)
            lab_acl = labels[:, 0].to(device)
            lab_men = labels[:, 1].to(device)
            lab_abn = labels[:, 2].to(device)

            out_acl, out_men, out_abn = model(volumes.float())

            loss_acl = criterion_acl(out_acl, lab_acl)
            loss_men = criterion_meniscus(out_men, lab_men)
            loss_abn = criterion_abnormal(out_abn, lab_abn)

            loss = (TASK_WEIGHT_ACL * loss_acl +
                    TASK_WEIGHT_MENISCUS * loss_men +
                    TASK_WEIGHT_ABNORMAL * loss_abn)
            total_loss += loss.item()

            probs_acl = torch.softmax(out_acl, dim=1)[:, 1].cpu().numpy()
            probs_men = torch.softmax(out_men, dim=1)[:, 1].cpu().numpy()
            probs_abn = torch.softmax(out_abn, dim=1)[:, 1].cpu().numpy()

            all_labels['acl'].extend(lab_acl.cpu().numpy())
            all_labels['meniscus'].extend(lab_men.cpu().numpy())
            all_labels['abnormal'].extend(lab_abn.cpu().numpy())
            all_probs['acl'].extend(probs_acl)
            all_probs['meniscus'].extend(probs_men)
            all_probs['abnormal'].extend(probs_abn)

    avg_loss = total_loss / len(loader)
    aucs = {}
    for task in ['acl', 'meniscus', 'abnormal']:
        try:
            aucs[task] = roc_auc_score(all_labels[task], all_probs[task])
        except ValueError:
            aucs[task] = 0.5

    return avg_loss, aucs, all_labels, all_probs

# %% [markdown]
# # Cell 10: Training Loop

# %%
history = {'train_loss': [], 'val_loss': [],
           'train_auc_acl': [], 'val_auc_acl': [],
           'train_auc_men': [], 'val_auc_men': [],
           'train_auc_abn': [], 'val_auc_abn': [],
           'lr': []}

best_composite_auc = 0.0
no_improve = 0
SAVE_PATH = '/content/drive/MyDrive/dataset/best_acl_model_v11.pth'

print(f"Training for up to {NUM_EPOCHS} epochs (patience={PATIENCE})...")
print(f"All backbone trainable + composite model selection\n")

for epoch in range(NUM_EPOCHS):
    current_lr = optimizer.param_groups[0]['lr']

    train_loss, train_aucs = train_epoch(model, train_loader, optimizer, device)
    val_loss, val_aucs, val_labels, val_probs = validate(model, val_loader, device)

    # Log history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_auc_acl'].append(train_aucs['acl'])
    history['val_auc_acl'].append(val_aucs['acl'])
    history['train_auc_men'].append(train_aucs['meniscus'])
    history['val_auc_men'].append(val_aucs['meniscus'])
    history['train_auc_abn'].append(train_aucs['abnormal'])
    history['val_auc_abn'].append(val_aucs['abnormal'])
    history['lr'].append(current_lr)

    # Composite AUC for model selection
    composite_val = (SEL_WEIGHT_ACL * val_aucs['acl'] +
                     SEL_WEIGHT_MEN * val_aucs['meniscus'] +
                     SEL_WEIGHT_ABN * val_aucs['abnormal'])
    composite_train = (SEL_WEIGHT_ACL * train_aucs['acl'] +
                       SEL_WEIGHT_MEN * train_aucs['meniscus'] +
                       SEL_WEIGHT_ABN * train_aucs['abnormal'])

    # Step scheduler on composite val AUC
    scheduler.step(composite_val)

    gap = 100 * (composite_train - composite_val)
    overfit_flag = " OVERFIT" if gap > 10 else ""

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}  (lr={current_lr:.1e})")
    print(f"  Train: loss={train_loss:.4f}  ACL={train_aucs['acl']:.4f}  Men={train_aucs['meniscus']:.4f}  Abn={train_aucs['abnormal']:.4f}")
    print(f"  Val:   loss={val_loss:.4f}  ACL={val_aucs['acl']:.4f}  Men={val_aucs['meniscus']:.4f}  Abn={val_aucs['abnormal']:.4f}  (gap={gap:.1f}%){overfit_flag}")
    print(f"  Composite AUC: train={composite_train:.4f}  val={composite_val:.4f}")

    # Model selection on composite AUC
    if composite_val > best_composite_auc:
        best_composite_auc = composite_val
        no_improve = 0
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  -> Saved best (Composite={best_composite_auc:.4f}, ACL={val_aucs['acl']:.4f}, Men={val_aucs['meniscus']:.4f}, Abn={val_aucs['abnormal']:.4f})")
    else:
        no_improve += 1
        print(f"  -> No improvement ({no_improve}/{PATIENCE})")
        if no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

print(f"\nBest composite val AUC: {best_composite_auc:.4f}")

# %% [markdown]
# # Cell 11: Training History Plots

# %%
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
epochs_range = range(1, len(history['train_loss']) + 1)

# Loss
axes[0].plot(epochs_range, history['train_loss'], 'b-', label='Train')
axes[0].plot(epochs_range, history['val_loss'], 'r-', label='Val')
axes[0].set_title('Total Loss', fontsize=14)
axes[0].legend()
axes[0].set_ylabel('Loss')

# AUCs
axes[1].plot(epochs_range, history['train_auc_acl'], 'b-', label='Train ACL')
axes[1].plot(epochs_range, history['val_auc_acl'], 'r-', label='Val ACL')
axes[1].plot(epochs_range, history['train_auc_men'], 'b--', alpha=0.5, label='Train Men')
axes[1].plot(epochs_range, history['val_auc_men'], 'r--', alpha=0.5, label='Val Men')
axes[1].plot(epochs_range, history['train_auc_abn'], 'b:', alpha=0.5, label='Train Abn')
axes[1].plot(epochs_range, history['val_auc_abn'], 'r:', alpha=0.5, label='Val Abn')
axes[1].set_title('AUC by Task', fontsize=14)
axes[1].legend(fontsize=7)
axes[1].set_ylabel('AUC')
axes[1].set_ylim(0.4, 1.0)

# LR
axes[2].plot(epochs_range, history['lr'], 'g-')
axes[2].set_title('Learning Rate', fontsize=14)
axes[2].set_yscale('log')
axes[2].set_ylabel('LR')

for ax in axes:
    ax.set_xlabel('Epoch')

plt.tight_layout()
plt.savefig('/content/drive/MyDrive/dataset/training_history_v11.png', dpi=150)
plt.show()

# %% [markdown]
# # Cell 12: Final Evaluation (All Tasks)

# %%
model.load_state_dict(torch.load(SAVE_PATH, map_location=device, weights_only=True))
model.eval()

val_loss, val_aucs, val_labels, val_probs = validate(model, val_loader, device)

# Find optimal thresholds for each task
for task, title in [('acl', 'ACL'), ('meniscus', 'Meniscus'), ('abnormal', 'Abnormal')]:
    fpr, tpr, thresholds = roc_curve(val_labels[task], val_probs[task])
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    best_thr = thresholds[best_idx]
    print(f"{title}: Optimal threshold={best_thr:.4f} (J={youden_j[best_idx]:.4f}, Sens={tpr[best_idx]:.3f}, Spec={1-fpr[best_idx]:.3f})")

label_names = ['Normal', 'Tear']

print('\n' + '=' * 60)
print('RESULTS — V11 (Multi-Task MRNet)')
print('=' * 60)
print(f'ACL AUC:      {val_aucs["acl"]:.4f}')
print(f'Meniscus AUC: {val_aucs["meniscus"]:.4f}')
print(f'Abnormal AUC: {val_aucs["abnormal"]:.4f}')
composite = SEL_WEIGHT_ACL*val_aucs["acl"] + SEL_WEIGHT_MEN*val_aucs["meniscus"] + SEL_WEIGHT_ABN*val_aucs["abnormal"]
print(f'Composite:    {composite:.4f}')

for task, title in [('acl', 'ACL'), ('meniscus', 'Meniscus'), ('abnormal', 'Abnormal')]:
    print(f'\n--- {title}: Default threshold (0.5) ---')
    preds = [1 if p >= 0.5 else 0 for p in val_probs[task]]
    print(classification_report(val_labels[task], preds, target_names=label_names, digits=3))

# %% [markdown]
# # Cell 13: Confusion Matrices (All Tasks)

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, task, title in zip(axes,
    ['acl', 'meniscus', 'abnormal'],
    ['ACL Tear', 'Meniscus Tear', 'Abnormality']):
    preds = [1 if p >= 0.5 else 0 for p in val_probs[task]]
    cm = confusion_matrix(val_labels[task], preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Positive'], yticklabels=['Normal', 'Positive'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'{title} (AUC={val_aucs[task]:.3f})')

plt.suptitle('V11 Multi-Task — Confusion Matrices', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/dataset/confusion_matrix_v11.png', dpi=150)
plt.show()
