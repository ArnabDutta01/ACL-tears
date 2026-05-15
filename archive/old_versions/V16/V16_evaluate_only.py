# ============================================================
# V16 Evaluation-Only Script
# Loads saved V16 .pth models from Drive, runs validation,
# and outputs the full results matrix + confusion matrices.
# NO TRAINING REQUIRED.
# ============================================================

# Cell 1: Mount Drive
from google.colab import drive
drive.mount("/content/drive")

# Cell 2: Config
DATA_DIR = '/content/drive/MyDrive/dataset/mrnet_all'
BATCH_SIZE = 1
RANDOM_SEED = 42
MAX_SLICES = 40
DROPOUT = 0.3
TASK_WEIGHT_ACL = 1.0
TASK_WEIGHT_MENISCUS = 1.0
TASK_WEIGHT_ABNORMAL = 0.5
SEL_WEIGHT_ACL = 0.5
SEL_WEIGHT_MEN = 0.3
SEL_WEIGHT_ABN = 0.2
ALL_VIEWS = ['sagittal', 'coronal', 'axial']

# Cell 3: Imports + Setup
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
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

# Cell 4: Load metadata + reproduce the exact same train/val split
data_path = Path(DATA_DIR)
metadata = pd.read_csv(data_path / 'metadata.csv')
print(f"Total patients: {len(metadata)}")

train_df, val_df = train_test_split(
    metadata, test_size=0.15, random_state=RANDOM_SEED,
    stratify=metadata['label_acl']
)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
print(f"Train: {len(train_df)} | Val: {len(val_df)}")

# Cell 5: Dataset class
class SingleViewDataset(Dataset):
    def __init__(self, df, data_dir, view='sagittal', max_slices=MAX_SLICES, augment=False):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.view = view
        self.max_slices = max_slices
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
        volume = data[self.view]
        slices = volume.astype(np.float32) / 255.0
        if slices.shape[0] > self.max_slices:
            offset = (slices.shape[0] - self.max_slices) // 2
            slices = slices[offset:offset + self.max_slices]
        slices_3ch = np.stack((slices,)*3, axis=1)
        tensor = torch.FloatTensor(slices_3ch)
        labels = torch.LongTensor([
            int(row['label_acl']),
            int(row['label_meniscus']),
            int(row['label_abnormal'])
        ])
        return tensor, labels

# Cell 6: Compute class weights from training set
_tmp = SingleViewDataset(train_df, DATA_DIR, view='sagittal')

def compute_weight(df, col, valid_indices):
    labels = [int(df.iloc[i][col]) for i in valid_indices]
    n = len(labels)
    n_pos = sum(labels)
    n_neg = n - n_pos
    w_neg = n / (2 * n_neg) if n_neg > 0 else 1.0
    w_pos = n / (2 * n_pos) if n_pos > 0 else 1.0
    return torch.FloatTensor([w_neg, w_pos]).to(device)

weight_acl = compute_weight(train_df, 'label_acl', _tmp.valid_indices)
weight_meniscus = compute_weight(train_df, 'label_meniscus', _tmp.valid_indices)
weight_abnormal = compute_weight(train_df, 'label_abnormal', _tmp.valid_indices)

print(f"Class weights:")
print(f"  ACL:      Normal={weight_acl[0]:.3f}, Tear={weight_acl[1]:.3f}")
print(f"  Meniscus: Normal={weight_meniscus[0]:.3f}, Tear={weight_meniscus[1]:.3f}")
print(f"  Abnormal: Normal={weight_abnormal[0]:.3f}, Abnormal={weight_abnormal[1]:.3f}")

# Cell 7: V16 Model architecture (EfficientNet-B0 + max-pool)
class MRNetPerView(nn.Module):
    def __init__(self, dropout=DROPOUT):
        super().__init__()
        backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.features = backbone.features
        self.pool = backbone.avgpool
        self.drop = nn.Dropout(p=dropout)
        self.head_acl = nn.Linear(1280, 2)
        self.head_meniscus = nn.Linear(1280, 2)
        self.head_abnormal = nn.Linear(1280, 2)

    def forward(self, x):
        x = x.squeeze(0)
        features = self.features(x)
        pooled = self.pool(features).flatten(1)
        volume_feat = torch.max(pooled, 0, keepdim=True)[0]
        volume_feat = self.drop(volume_feat)
        return (self.head_acl(volume_feat),
                self.head_meniscus(volume_feat),
                self.head_abnormal(volume_feat))

# Cell 8: Validate function
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

# Cell 9: Load all 3 view models and evaluate
print("\n" + "="*60)
print("  LOADING & EVALUATING V16 MODELS")
print("="*60)

view_results = {}
crits = [
    nn.CrossEntropyLoss(weight=weight_acl),
    nn.CrossEntropyLoss(weight=weight_meniscus),
    nn.CrossEntropyLoss(weight=weight_abnormal),
]

for view in ALL_VIEWS:
    save_path = f'/content/drive/MyDrive/dataset/best_v16_{view}.pth'
    print(f"\n--- {view.upper()} ---")
    print(f"  Loading: {save_path}")

    val_dataset = SingleViewDataset(val_df, DATA_DIR, view=view, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)

    model = MRNetPerView(dropout=DROPOUT).to(device)
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))

    _, _, val_labels, val_probs = validate(model, val_loader, crits, device)
    view_results[view] = {
        'labels': val_labels,
        'probs': val_probs,
        'aucs': {t: roc_auc_score(val_labels[t], val_probs[t])
                 for t in ['acl', 'meniscus', 'abnormal']}
    }
    a = view_results[view]['aucs']
    print(f"  ACL={a['acl']:.4f}  Men={a['meniscus']:.4f}  Abn={a['abnormal']:.4f}")
    del model
    torch.cuda.empty_cache()

# Cell 10: Combine views + output matrix
print("\n" + "="*60)
print("  COMBINING VIEWS (MRNet-style)")
print("="*60)

for view in ALL_VIEWS:
    a = view_results[view]['aucs']
    print(f"  {view:10s}: ACL={a['acl']:.4f}  Men={a['meniscus']:.4f}  Abn={a['abnormal']:.4f}")

# Simple average
print("\n--- Method 1: Simple Average ---")
combined_probs = {}
for task in ['acl', 'meniscus', 'abnormal']:
    combined_probs[task] = np.mean(
        [view_results[v]['probs'][task] for v in ALL_VIEWS], axis=0
    )
    labels = view_results[ALL_VIEWS[0]]['labels'][task]
    auc = roc_auc_score(labels, combined_probs[task])
    print(f"  {task:10s} AUC = {auc:.4f}")

composite = (SEL_WEIGHT_ACL * roc_auc_score(view_results[ALL_VIEWS[0]]['labels']['acl'], combined_probs['acl']) +
             SEL_WEIGHT_MEN * roc_auc_score(view_results[ALL_VIEWS[0]]['labels']['meniscus'], combined_probs['meniscus']) +
             SEL_WEIGHT_ABN * roc_auc_score(view_results[ALL_VIEWS[0]]['labels']['abnormal'], combined_probs['abnormal']))
print(f"  Composite = {composite:.4f}")

# Logistic regression
print("\n--- Method 2: Logistic Regression (MRNet paper) ---")
for task in ['acl', 'meniscus', 'abnormal']:
    X = np.column_stack([view_results[v]['probs'][task] for v in ALL_VIEWS])
    y = np.array(view_results[ALL_VIEWS[0]]['labels'][task])
    lr_model = LogisticRegression(random_state=RANDOM_SEED)
    lr_model.fit(X, y)
    lr_probs = lr_model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, lr_probs)
    print(f"  {task:10s} AUC = {auc:.4f}  (weights: {', '.join(f'{v}={w:.3f}' for v, w in zip(ALL_VIEWS, lr_model.coef_[0]))})")

# Cell 11: Classification reports + confusion matrices
print('\n' + '=' * 60)
print('RESULTS — V16 MRNet (Per-View Multi-Task)')
print('=' * 60)

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
print("\nDone! Simple Average confusion matrix saved to Drive.")

# Cell 12: Logistic Regression combined confusion matrices
print('\n' + '=' * 60)
print('RESULTS — V16 Logistic Regression Combined')
print('=' * 60)

lr_combined_probs = {}
for task in ['acl', 'meniscus', 'abnormal']:
    X = np.column_stack([view_results[v]['probs'][task] for v in ALL_VIEWS])
    y = np.array(view_results[ALL_VIEWS[0]]['labels'][task])
    lr_model = LogisticRegression(random_state=RANDOM_SEED)
    lr_model.fit(X, y)
    lr_combined_probs[task] = lr_model.predict_proba(X)[:, 1]

for task, title in [('acl', 'ACL'), ('meniscus', 'Meniscus'), ('abnormal', 'Abnormal')]:
    labels = view_results[ALL_VIEWS[0]]['labels'][task]
    probs = lr_combined_probs[task]
    auc = roc_auc_score(labels, probs)
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j = tpr - fpr
    best_idx = np.argmax(j)
    print(f"\n{title}: AUC={auc:.4f}, Threshold={thresholds[best_idx]:.4f}, "
          f"Sens={tpr[best_idx]:.3f}, Spec={1-fpr[best_idx]:.3f}")
    preds = [1 if p >= 0.5 else 0 for p in probs]
    print(classification_report(labels, preds, target_names=label_names, digits=3))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, task, title in zip(axes, ['acl', 'meniscus', 'abnormal'],
                            ['ACL Tear', 'Meniscus Tear', 'Abnormality']):
    labels = view_results[ALL_VIEWS[0]]['labels'][task]
    probs = lr_combined_probs[task]
    preds = [1 if p >= 0.5 else 0 for p in probs]
    cm = confusion_matrix(labels, preds)
    auc = roc_auc_score(labels, probs)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Normal', 'Positive'], yticklabels=['Normal', 'Positive'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'{title} (AUC={auc:.3f})')

plt.suptitle('V16 MRNet — Logistic Regression Combined Confusion Matrices', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/dataset/confusion_matrix_v16_lr.png', dpi=150)
plt.show()
print("\nDone! LR confusion matrix saved to Drive.")
