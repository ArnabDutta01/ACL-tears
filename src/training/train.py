# ============================================================
# ACL Tear Detection — V16 Faithful MRNet Multi-Task (HPC)
# ============================================================
# Adapted from the Colab V16 notebook for HPC/SLURM execution.
#
# Usage:
#   python train_v16.py --data_dir ./data/mrnet_all --output_dir ./outputs
# ============================================================

import argparse
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
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm.auto import tqdm
import warnings
import os

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='V16 MRNet Multi-Task Training')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to mrnet_all directory')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Directory for saving checkpoints and plots')
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--max_slices', type=int, default=40)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--views', nargs='+', default=['sagittal', 'coronal', 'axial'])
    return parser.parse_args()


# ============================================================
# Dataset
# ============================================================
class SingleViewDataset(Dataset):
    """Loads ONE view at a time from mrnet_all .npz files with 3 labels."""
    def __init__(self, df, data_dir, view='sagittal', max_slices=40, augment=False):
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
        volume = data[self.view]
        slices = volume.astype(np.float32) / 255.0

        if slices.shape[0] > self.max_slices:
            offset = (slices.shape[0] - self.max_slices) // 2
            slices = slices[offset:offset + self.max_slices]

        if self.augment and np.random.random() > 0.5:
            slices = slices[:, :, ::-1].copy()

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


# ============================================================
# Model
# ============================================================
class MRNetPerView(nn.Module):
    """Multi-task MRNet for a SINGLE view (identical to V11/V16)."""
    def __init__(self, dropout=0.3):
        super().__init__()
        backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.features = backbone.features
        self.pool = backbone.avgpool
        self.drop = nn.Dropout(p=dropout)
        self.head_acl = nn.Linear(1280, 2)
        self.head_meniscus = nn.Linear(1280, 2)
        self.head_abnormal = nn.Linear(1280, 2)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Backbone: EfficientNet-B0 (all trainable)")
        print(f"  Heads: ACL(1280->2), Men(1280->2), Abn(1280->2)")
        print(f"  Params: {trainable:,} trainable / {total:,} total")

    def forward(self, x):
        x = x.squeeze(0)
        features = self.features(x)
        pooled = self.pool(features).flatten(1)
        volume_feat = torch.max(pooled, 0, keepdim=True)[0]
        volume_feat = self.drop(volume_feat)
        return (self.head_acl(volume_feat),
                self.head_meniscus(volume_feat),
                self.head_abnormal(volume_feat))


# ============================================================
# Training & Validation
# ============================================================
TASK_WEIGHT_ACL = 1.0
TASK_WEIGHT_MENISCUS = 1.0
TASK_WEIGHT_ABNORMAL = 0.5
SEL_WEIGHT_ACL = 0.5
SEL_WEIGHT_MEN = 0.3
SEL_WEIGHT_ABN = 0.2


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


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')

    # Load metadata + split
    data_path = Path(args.data_dir)
    metadata = pd.read_csv(data_path / 'metadata.csv')

    print(f"\nTotal patients: {len(metadata)}")
    print(f"  ACL tear:      {metadata.label_acl.sum():4d} / {len(metadata)} ({100*metadata.label_acl.mean():.1f}%)")
    print(f"  Meniscus tear: {metadata.label_meniscus.sum():4d} / {len(metadata)} ({100*metadata.label_meniscus.mean():.1f}%)")
    print(f"  Abnormal:      {metadata.label_abnormal.sum():4d} / {len(metadata)} ({100*metadata.label_abnormal.mean():.1f}%)")

    train_df, val_df = train_test_split(
        metadata, test_size=0.15, random_state=args.seed,
        stratify=metadata['label_acl']
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print(f"\nTrain: {len(train_df)} patients (ACL tear: {train_df.label_acl.sum()})")
    print(f"Val:   {len(val_df)} patients (ACL tear: {val_df.label_acl.sum()})")

    # Compute class weights
    _tmp = SingleViewDataset(train_df, args.data_dir, view='sagittal', max_slices=args.max_slices)
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

    print(f"\nClass weights:")
    print(f"  ACL:      Normal={weight_acl[0]:.3f}, Tear={weight_acl[1]:.3f}")
    print(f"  Meniscus: Normal={weight_meniscus[0]:.3f}, Tear={weight_meniscus[1]:.3f}")
    print(f"  Abnormal: Normal={weight_abnormal[0]:.3f}, Abnormal={weight_abnormal[1]:.3f}")

    # ---- Train all view models ----
    view_results = {}

    for view in args.views:
        print(f"\n{'='*60}")
        print(f"  TRAINING VIEW: {view.upper()}")
        print(f"{'='*60}")

        train_dataset = SingleViewDataset(train_df, args.data_dir, view=view,
                                          max_slices=args.max_slices, augment=True)
        val_dataset = SingleViewDataset(val_df, args.data_dir, view=view,
                                        max_slices=args.max_slices, augment=False)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)

        print("Creating model...")
        model = MRNetPerView(dropout=args.dropout).to(device)
        crits = [
            nn.CrossEntropyLoss(weight=weight_acl),
            nn.CrossEntropyLoss(weight=weight_meniscus),
            nn.CrossEntropyLoss(weight=weight_abnormal),
        ]
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.3, patience=5, min_lr=1e-7)

        best_composite = 0.0
        no_improve = 0
        save_path = os.path.join(args.output_dir, f'best_v16_{view}.pth')

        for epoch in range(args.epochs):
            lr = optimizer.param_groups[0]['lr']
            train_loss, train_aucs = train_epoch(model, train_loader, optimizer, crits, device)
            val_loss, val_aucs, val_labels, val_probs = validate(model, val_loader, crits, device)

            composite_val = (SEL_WEIGHT_ACL * val_aucs['acl'] +
                             SEL_WEIGHT_MEN * val_aucs['meniscus'] +
                             SEL_WEIGHT_ABN * val_aucs['abnormal'])
            scheduler.step(composite_val)

            gap = 100 * ((SEL_WEIGHT_ACL * train_aucs['acl'] + SEL_WEIGHT_MEN * train_aucs['meniscus'] +
                           SEL_WEIGHT_ABN * train_aucs['abnormal']) - composite_val)

            print(f"[{view}] Epoch {epoch+1}/{args.epochs}  (lr={lr:.1e})")
            print(f"  Train: loss={train_loss:.4f}  ACL={train_aucs['acl']:.4f}  Men={train_aucs['meniscus']:.4f}  Abn={train_aucs['abnormal']:.4f}")
            print(f"  Val:   loss={val_loss:.4f}  ACL={val_aucs['acl']:.4f}  Men={val_aucs['meniscus']:.4f}  Abn={val_aucs['abnormal']:.4f}  (gap={gap:.1f}%)")

            if composite_val > best_composite:
                best_composite = composite_val
                no_improve = 0
                torch.save(model.state_dict(), save_path)
                print(f"  -> Saved best {view} (Comp={best_composite:.4f})")
            else:
                no_improve += 1
                print(f"  -> No improvement ({no_improve}/{args.patience})")
                if no_improve >= args.patience:
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

    # ---- Combine Views ----
    ALL_VIEWS = ['sagittal', 'coronal', 'axial']

    print(f"\n{'='*60}")
    print("  COMBINING VIEWS (MRNet-style)")
    print(f"{'='*60}")

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

    # Logistic regression combination
    print("\n--- Method 2: Logistic Regression (MRNet paper) ---")
    for task in ['acl', 'meniscus', 'abnormal']:
        X = np.column_stack([view_results[v]['probs'][task] for v in ALL_VIEWS])
        y = np.array(view_results[ALL_VIEWS[0]]['labels'][task])
        lr_model = LogisticRegression(random_state=args.seed)
        lr_model.fit(X, y)
        lr_probs = lr_model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, lr_probs)
        print(f"  {task:10s} AUC = {auc:.4f}  (weights: {', '.join(f'{v}={w:.3f}' for v, w in zip(ALL_VIEWS, lr_model.coef_[0]))})")

    # ---- Final Results + Confusion Matrices ----
    print(f"\n{'='*60}")
    print('RESULTS — V16 MRNet (Per-View Multi-Task)')
    print(f"{'='*60}")

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
    cm_path = os.path.join(args.output_dir, 'confusion_matrix_v16.png')
    plt.savefig(cm_path, dpi=150)
    print(f"\nConfusion matrix saved to {cm_path}")
    print("\nDone!")


if __name__ == '__main__':
    main()
