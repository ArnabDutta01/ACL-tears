# ============================================================
# ACL Tear Detection — V16 Faithful MRNet Multi-Task
# ============================================================
# Follows the original MRNet paper (Bien et al. 2018):
#   - Separate model per VIEW (not one shared model)
#   - AlexNet backbone (paper-faithful) OR EfficientNet-B0
#   - Max-pool across slices → classifier
#   - Combine view predictions via learned logistic regression
#
# Extension over original MRNet:
#   - Multi-task heads (ACL, Meniscus, Abnormal) per model
#   - This means 3 models (one per view) instead of 9
#
# V11 baseline (sagittal only): ACL=0.923, Men=0.794, Abn=0.770
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
NUM_EPOCHS = 35
RANDOM_SEED = 42
MAX_SLICES = 40          # Single view at a time, same as V11
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

# Which views to train (set to subset for testing)
VIEWS = ['sagittal', 'coronal', 'axial']

# %% [markdown]
# # Cell 3: Imports

# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
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

# Stratified split on ACL label (primary task)
train_df, val_df = train_test_split(
    metadata, test_size=0.15, random_state=RANDOM_SEED,
    stratify=metadata['label_acl']
)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print(f"\nTrain: {len(train_df)} patients")
print(f"  ACL: {train_df.label_acl.sum()} tear / {(train_df.label_acl==0).sum()} normal")
print(f"Val: {len(val_df)} patients")
print(f"  ACL: {val_df.label_acl.sum()} tear / {(val_df.label_acl==0).sum()} normal")

# %% [markdown]
# # Cell 5: Single-View Dataset (like V11)

# %%
class SingleViewDataset(Dataset):
    """Loads ONE view at a time from mrnet_all .npz files with 3 labels.
    This is the MRNet-faithful approach: one model per view.
    """
    def __init__(self, df, data_dir, view='sagittal', max_slices=MAX_SLICES, augment=False):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.view = view
        self.max_slices = max_slices
        self.augment = augment
        self.valid_indices = []
        for idx in range(len(self.df)):
            fpath = self.data_dir / self.df.iloc[idx]['filename']
            if fpath.exists():
                self.valid_indices.append(idx)
        print(f"  [{view}] {len(self.valid_indices)} valid patients (of {len(self.df)})")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        patient_idx = self.valid_indices[idx]
        row = self.df.iloc[patient_idx]
        data = np.load(self.data_dir / row['filename'])
        volume = data[self.view]  # (S, 256, 256), uint8
        slices = volume.astype(np.float32) / 255.0

        # Center crop if too many slices
        if slices.shape[0] > self.max_slices:
            offset = (slices.shape[0] - self.max_slices) // 2
            slices = slices[offset:offset + self.max_slices]

        # Simple augmentation: random horizontal flip
        if self.augment and np.random.random() > 0.5:
            slices = slices[:, :, ::-1].copy()

        # Grayscale -> 3-channel: (S, H, W) -> (S, 3, H, W)
        slices_3ch = np.stack((slices,)*3, axis=1)
        tensor = torch.FloatTensor(slices_3ch)

        labels = torch.LongTensor([
            int(row['label_acl']),
            int(row['label_meniscus']),
            int(row['label_abnormal'])
        ])
        return tensor, labels

    def get_label_counts(self):
        acl = [int(self.df.iloc[i]['label_acl']) for i in self.valid_indices]
        men = [int(self.df.iloc[i]['label_meniscus']) for i in self.valid_indices]
        abn = [int(self.df.iloc[i]['label_abnormal']) for i in self.valid_indices]
        return {'acl': acl, 'meniscus': men, 'abnormal': abn}

# %% [markdown]
# # Cell 6: Compute Class Weights

# %%
# Use sagittal dataset for label counts (same labels regardless of view)
_tmp = SingleViewDataset(train_df, DATA_DIR, view='sagittal')
label_counts = _tmp.get_label_counts()

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
# # Cell 7: Model — V11 Architecture (proven, per-view)

# %%
class MRNetPerView(nn.Module):
    """Multi-task MRNet for a SINGLE view.
    Identical to V11 which achieved ACL=0.923.
    One instance trained per view (sagittal, coronal, axial).
    """
    def __init__(self, dropout=DROPOUT):
        super().__init__()
        backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.features = backbone.features
        self.pool = backbone.avgpool
        self.drop = nn.Dropout(p=dropout)

        # 3 task-specific heads (same as V11)
        self.head_acl = nn.Linear(1280, 2)
        self.head_meniscus = nn.Linear(1280, 2)
        self.head_abnormal = nn.Linear(1280, 2)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Backbone: EfficientNet-B0 (all trainable)")
        print(f"  Heads: ACL(1280->2), Men(1280->2), Abn(1280->2)")
        print(f"  Params: {trainable:,} trainable / {total:,} total")

    def forward(self, x):
        x = x.squeeze(0)            # (S, 3, H, W)
        features = self.features(x)  # (S, 1280, 8, 8)
        pooled = self.pool(features).flatten(1)  # (S, 1280)
        volume_feat = torch.max(pooled, 0, keepdim=True)[0]  # (1, 1280)
        volume_feat = self.drop(volume_feat)
        return (self.head_acl(volume_feat),
                self.head_meniscus(volume_feat),
                self.head_abnormal(volume_feat))

# %% [markdown]
# # Cell 8: Train + Validate Functions (same as V11)

# %%
def train_epoch(model, loader, optimizer, crits, device):
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

        loss = (TASK_WEIGHT_ACL * crits[0](out_acl, lab_acl) +
                TASK_WEIGHT_MENISCUS * crits[1](out_men, lab_men) +
                TASK_WEIGHT_ABNORMAL * crits[2](out_abn, lab_abn))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        all_labels['acl'].extend(lab_acl.cpu().numpy())
        all_labels['meniscus'].extend(lab_men.cpu().numpy())
        all_labels['abnormal'].extend(lab_abn.cpu().numpy())
        all_probs['acl'].extend(torch.softmax(out_acl, 1)[:, 1].detach().cpu().numpy())
        all_probs['meniscus'].extend(torch.softmax(out_men, 1)[:, 1].detach().cpu().numpy())
        all_probs['abnormal'].extend(torch.softmax(out_abn, 1)[:, 1].detach().cpu().numpy())

    avg_loss = total_loss / len(loader)
    aucs = {}
    for task in ['acl', 'meniscus', 'abnormal']:
        try:
            aucs[task] = roc_auc_score(all_labels[task], all_probs[task])
        except ValueError:
            aucs[task] = 0.5
    return avg_loss, aucs


def validate(model, loader, crits, device):
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
            loss = (TASK_WEIGHT_ACL * crits[0](out_acl, lab_acl) +
                    TASK_WEIGHT_MENISCUS * crits[1](out_men, lab_men) +
                    TASK_WEIGHT_ABNORMAL * crits[2](out_abn, lab_abn))
            total_loss += loss.item()

            all_labels['acl'].extend(lab_acl.cpu().numpy())
            all_labels['meniscus'].extend(lab_men.cpu().numpy())
            all_labels['abnormal'].extend(lab_abn.cpu().numpy())
            all_probs['acl'].extend(torch.softmax(out_acl, 1)[:, 1].cpu().numpy())
            all_probs['meniscus'].extend(torch.softmax(out_men, 1)[:, 1].cpu().numpy())
            all_probs['abnormal'].extend(torch.softmax(out_abn, 1)[:, 1].cpu().numpy())

    avg_loss = total_loss / len(loader)
    aucs = {}
    for task in ['acl', 'meniscus', 'abnormal']:
        try:
            aucs[task] = roc_auc_score(all_labels[task], all_probs[task])
        except ValueError:
            aucs[task] = 0.5
    return avg_loss, aucs, all_labels, all_probs

# %% [markdown]
# # Cell 9: Train All 3 View Models

# %%
view_results = {}  # Store per-view val predictions

for view in VIEWS:
    print(f"\n{'='*60}")
    print(f"  TRAINING VIEW: {view.upper()}")
    print(f"{'='*60}")

    # Create datasets for this view
    print("Creating datasets...")
    train_dataset = SingleViewDataset(train_df, DATA_DIR, view=view, augment=True)
    val_dataset = SingleViewDataset(val_df, DATA_DIR, view=view, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)

    # Create model, optimizer, scheduler
    print("Creating model...")
    model = MRNetPerView(dropout=DROPOUT).to(device)
    crits = [
        nn.CrossEntropyLoss(weight=weight_acl),
        nn.CrossEntropyLoss(weight=weight_meniscus),
        nn.CrossEntropyLoss(weight=weight_abnormal),
    ]
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.3, patience=5, min_lr=1e-7)

    # Training loop
    best_composite = 0.0
    no_improve = 0
    save_path = f'/content/drive/MyDrive/dataset/best_v16_{view}.pth'

    for epoch in range(NUM_EPOCHS):
        lr = optimizer.param_groups[0]['lr']
        train_loss, train_aucs = train_epoch(model, train_loader, optimizer, crits, device)
        val_loss, val_aucs, val_labels, val_probs = validate(model, val_loader, crits, device)

        composite_val = (SEL_WEIGHT_ACL * val_aucs['acl'] +
                         SEL_WEIGHT_MEN * val_aucs['meniscus'] +
                         SEL_WEIGHT_ABN * val_aucs['abnormal'])
        scheduler.step(composite_val)

        gap = 100 * ((SEL_WEIGHT_ACL * train_aucs['acl'] + SEL_WEIGHT_MEN * train_aucs['meniscus'] +
                       SEL_WEIGHT_ABN * train_aucs['abnormal']) - composite_val)

        print(f"[{view}] Epoch {epoch+1}/{NUM_EPOCHS}  (lr={lr:.1e})")
        print(f"  Train: loss={train_loss:.4f}  ACL={train_aucs['acl']:.4f}  Men={train_aucs['meniscus']:.4f}  Abn={train_aucs['abnormal']:.4f}")
        print(f"  Val:   loss={val_loss:.4f}  ACL={val_aucs['acl']:.4f}  Men={val_aucs['meniscus']:.4f}  Abn={val_aucs['abnormal']:.4f}  (gap={gap:.1f}%)")

        if composite_val > best_composite:
            best_composite = composite_val
            no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best {view} (Comp={best_composite:.4f})")
        else:
            no_improve += 1
            print(f"  -> No improvement ({no_improve}/{PATIENCE})")
            if no_improve >= PATIENCE:
                print(f"  Early stopping {view} at epoch {epoch+1}")
                break

    # Load best and collect val predictions
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    _, _, val_labels, val_probs = validate(model, val_loader, crits, device)
    view_results[view] = {'labels': val_labels, 'probs': val_probs, 'aucs': {
        t: roc_auc_score(val_labels[t], val_probs[t]) for t in ['acl', 'meniscus', 'abnormal']
    }}
    print(f"\n  Best {view}: ACL={view_results[view]['aucs']['acl']:.4f}, "
          f"Men={view_results[view]['aucs']['meniscus']:.4f}, "
          f"Abn={view_results[view]['aucs']['abnormal']:.4f}")

    del model, optimizer, scheduler
    torch.cuda.empty_cache()

# %% [markdown]
# # Cell 10: Combine Views (MRNet-style logistic regression)

# %%
# Always use all 3 views for combination (VIEWS may have been filtered during resume)
ALL_VIEWS = ['sagittal', 'coronal', 'axial']

# Auto-load models if view_results is missing (e.g. resumed session)
if not view_results:
    print("view_results empty — loading saved models from Drive...\n")
    crits = [
        nn.CrossEntropyLoss(weight=weight_acl),
        nn.CrossEntropyLoss(weight=weight_meniscus),
        nn.CrossEntropyLoss(weight=weight_abnormal),
    ]
    for view in ALL_VIEWS:
        save_path = f'/content/drive/MyDrive/dataset/best_v16_{view}.pth'
        print(f"  Loading {view}...")
        val_dataset = SingleViewDataset(val_df, DATA_DIR, view=view, augment=False)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=2, pin_memory=True)
        model = MRNetPerView(dropout=DROPOUT).to(device)
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        _, _, val_labels, val_probs = validate(model, val_loader, crits, device)
        view_results[view] = {
            'labels': val_labels, 'probs': val_probs,
            'aucs': {t: roc_auc_score(val_labels[t], val_probs[t])
                     for t in ['acl', 'meniscus', 'abnormal']}
        }
        del model
        torch.cuda.empty_cache()
    print()

print("\\n" + "="*60)
print("  COMBINING VIEWS (MRNet-style)")
print("="*60)

# Per-view AUC summary
for view in ALL_VIEWS:
    a = view_results[view]['aucs']
    print(f"  {view:10s}: ACL={a['acl']:.4f}  Men={a['meniscus']:.4f}  Abn={a['abnormal']:.4f}")

# Simple average combination (no extra training needed)
print("\n--- Method 1: Simple Average ---")
combined_probs = {}
for task in ['acl', 'meniscus', 'abnormal']:
    combined_probs[task] = np.mean(
        [view_results[v]['probs'][task] for v in ALL_VIEWS], axis=0
    )
    labels = view_results[ALL_VIEWS[0]]['labels'][task]
    auc = roc_auc_score(labels, combined_probs[task])
    print(f"  {task:10s} AUC = {auc:.4f}")

composite = (SEL_WEIGHT_ACL * roc_auc_score(labels, combined_probs['acl']) +
             SEL_WEIGHT_MEN * roc_auc_score(view_results[ALL_VIEWS[0]]['labels']['meniscus'], combined_probs['meniscus']) +
             SEL_WEIGHT_ABN * roc_auc_score(view_results[ALL_VIEWS[0]]['labels']['abnormal'], combined_probs['abnormal']))
print(f"  Composite = {composite:.4f}")

# Learned combination via logistic regression (MRNet paper approach)
print("\n--- Method 2: Logistic Regression (MRNet paper) ---")
for task in ['acl', 'meniscus', 'abnormal']:
    X = np.column_stack([view_results[v]['probs'][task] for v in ALL_VIEWS])
    y = np.array(view_results[ALL_VIEWS[0]]['labels'][task])
    lr_model = LogisticRegression(random_state=RANDOM_SEED)
    lr_model.fit(X, y)
    lr_probs = lr_model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, lr_probs)
    print(f"  {task:10s} AUC = {auc:.4f}  (weights: {', '.join(f'{v}={w:.3f}' for v, w in zip(ALL_VIEWS, lr_model.coef_[0]))})")

# %% [markdown]
# # Cell 11: Final Results + Confusion Matrices

# %%
print('\n' + '=' * 60)
print('RESULTS — V16 MRNet (Per-View Multi-Task)')
print('=' * 60)

# Use simple average as final predictions
label_names = ['Normal', 'Tear']
for task, title in [('acl', 'ACL'), ('meniscus', 'Meniscus'), ('abnormal', 'Abnormal')]:
    labels = view_results[ALL_VIEWS[0]]['labels'][task]
    probs = combined_probs[task]
    auc = roc_auc_score(labels, probs)
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j = tpr - fpr
    best_idx = np.argmax(j)
    print(f"\n{title}: AUC={auc:.4f}, Threshold={thresholds[best_idx]:.4f}, "
          f"Sens={tpr[best_idx]:.3f}, Spec={1-fpr[best_idx]:.3f}")
    preds = [1 if p >= 0.5 else 0 for p in probs]
    print(classification_report(labels, preds, target_names=label_names, digits=3))

# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, task, title in zip(axes, ['acl', 'meniscus', 'abnormal'],
                            ['ACL Tear', 'Meniscus Tear', 'Abnormality']):
    labels = view_results[ALL_VIEWS[0]]['labels'][task]
    probs = combined_probs[task]
    preds = [1 if p >= 0.5 else 0 for p in probs]
    cm = confusion_matrix(labels, preds)
    auc = roc_auc_score(labels, probs)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Positive'], yticklabels=['Normal', 'Positive'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'{title} (AUC={auc:.3f})')

plt.suptitle('V16 MRNet Per-View Multi-Task — Confusion Matrices', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/dataset/confusion_matrix_v16.png', dpi=150)
plt.show()
