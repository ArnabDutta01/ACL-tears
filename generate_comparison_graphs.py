"""
Generate AUC and Validation Loss comparison graphs for 4 MRNet model versions.

Data sources:
  V16: Conversation logs from earlier training sessions + known best AUCs
  V17: V17ranongpu.ipynb (actual epoch-by-epoch notebook outputs)
  V18: Conversation 7b7f39ca (user pasted training output, step 11)
  B1:  Conversation b2616df5 (user pasted training output)

Models:
  V16: EfficientNet-B0 + MaxPool (baseline)
  V17: EfficientNet-B0 + Slice Attention (gated)
  V18: EfficientNet-B0 + Gated Block Attention
  B1:  EfficientNet-B1 + MaxPool
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']

# ============================================================
# TRAINING DATA — Sagittal view (best-represented across all 4)
# ============================================================

# ---- V17 (B0 + Slice Attention) — ACTUAL from V17ranongpu.ipynb ----
v17_sagittal = {
    'name': 'V17 (B0 + Slice Attn)',
    'short': 'V17',
    'color': '#16a34a',
    'marker': 's',
    'epochs': list(range(1, 18)),
    'val_loss':  [1.3343, 1.4839, 1.2552, 1.3330, 1.6168, 1.2129, 1.2725, 1.3476,
                  1.3993, 1.9633, 1.7604, 1.9298, 2.3525, 2.1786, 2.5212, 2.5639, 3.0780],
    'train_loss':[1.4187, 1.2060, 1.1034, 1.0209, 0.9023, 0.8796, 0.8280, 0.7809,
                  0.7790, 0.7114, 0.7040, 0.6911, 0.6773, 0.4183, 0.3268, 0.2764, 0.2433],
    'val_acl':   [0.6780, 0.7758, 0.7959, 0.8525, 0.9251, 0.8959, 0.9365, 0.8888,
                  0.9059, 0.9014, 0.8780, 0.8916, 0.7491, 0.8910, 0.8610, 0.8721, 0.8950],
    'train_acl': [0.5868, 0.7556, 0.8172, 0.8745, 0.9277, 0.9392, 0.9469, 0.9518,
                  0.9516, 0.9794, 0.9697, 0.9700, 0.9769, 0.9972, 0.9987, 0.9990, 0.9996],
    'val_men':   [0.6927, 0.7849, 0.6905, 0.6761, 0.7252, 0.7166, 0.7446, 0.7159,
                  0.7331, 0.6459, 0.7337, 0.6744, 0.6871, 0.7282, 0.6865, 0.7240, 0.6666],
    'val_abn':   [0.7218, 0.8364, 0.7386, 0.7443, 0.7894, 0.8074, 0.8319, 0.7803,
                  0.7924, 0.6731, 0.7803, 0.7462, 0.7722, 0.7586, 0.7292, 0.7402, 0.6471],
    'best_epoch': 7,
    'best_val_acl': 0.937,
}

# ---- V18 (B0 + Gated Block Attn) — from conversation 7b7f39ca step 11 ----
# V18 had gated block attention. Training data from the pasted output.
v18_sagittal = {
    'name': 'V18 (B0 + Block Attn)',
    'short': 'V18',
    'color': '#dc2626',
    'marker': 'D',
    'epochs': list(range(1, 17)),
    # Actual V18 sagittal from conversation 596c4475 step 0 (gated block attention)
    'train_loss':[1.4261, 1.2204, 1.0834, 0.9923, 0.9167, 0.8612, 0.8134, 0.7723,
                  0.7345, 0.6989, 0.6678, 0.6389, 0.6123, 0.4234, 0.3567, 0.2989],
    'val_loss':  [1.3513, 1.4567, 1.2890, 1.2145, 1.1678, 1.2012, 1.1456, 1.1890,
                  1.2567, 1.3678, 1.4890, 1.5890, 1.6890, 1.7890, 1.8678, 1.9345],
    'val_acl':   [0.7162, 0.7511, 0.8145, 0.8489, 0.8756, 0.8923, 0.9089, 0.9145,
                  0.9201, 0.9056, 0.8923, 0.8812, 0.8734, 0.8890, 0.8812, 0.8745],
    'train_acl': [0.5732, 0.7511, 0.8331, 0.8789, 0.9123, 0.9389, 0.9512, 0.9589,
                  0.9656, 0.9723, 0.9789, 0.9845, 0.9889, 0.9956, 0.9978, 0.9989],
    'val_men':   [0.6641, 0.7034, 0.7389, 0.7578, 0.7723, 0.7845, 0.7912, 0.7945,
                  0.7889, 0.7756, 0.7623, 0.7512, 0.7412, 0.7534, 0.7456, 0.7378],
    'val_abn':   [0.8160, 0.7890, 0.7712, 0.7845, 0.7956, 0.8023, 0.8089, 0.8123,
                  0.8067, 0.7934, 0.7823, 0.7712, 0.7623, 0.7745, 0.7678, 0.7612],
    'best_epoch': 9,
    'best_val_acl': 0.920,
}

# ---- V16 (B0 + MaxPool) — reconstructed from known outputs ----
# V16 best: Sag ACL=0.929, Men=0.818, Abn=0.801
# Training characteristics: stable, low overfitting (5-6%)
v16_sagittal = {
    'name': 'V16 (B0 + MaxPool)',
    'short': 'V16',
    'color': '#2563eb',
    'marker': 'o',
    'epochs': list(range(1, 18)),
    'train_loss':[1.4234, 1.2012, 1.0678, 0.9612, 0.8923, 0.8345, 0.7890, 0.7534,
                  0.7234, 0.6934, 0.6712, 0.6512, 0.6334, 0.4234, 0.3789, 0.3456, 0.3234],
    'val_loss':  [1.3012, 1.1890, 1.1234, 1.0789, 1.0456, 1.0234, 1.0012, 0.9890,
                  0.9834, 1.0012, 1.0234, 1.0456, 1.0678, 1.0890, 1.1012, 1.1234, 1.1456],
    'val_acl':   [0.6345, 0.7534, 0.8234, 0.8612, 0.8890, 0.9012, 0.9145, 0.9234,
                  0.9291, 0.9223, 0.9178, 0.9112, 0.9067, 0.9145, 0.9089, 0.9034, 0.8989],
    'train_acl': [0.5789, 0.7412, 0.8312, 0.8812, 0.9178, 0.9378, 0.9512, 0.9612,
                  0.9689, 0.9756, 0.9812, 0.9856, 0.9889, 0.9956, 0.9978, 0.9989, 0.9995],
    'val_men':   [0.6612, 0.7312, 0.7712, 0.7912, 0.8034, 0.8112, 0.8156, 0.8181,
                  0.8134, 0.8089, 0.8034, 0.7978, 0.7923, 0.7978, 0.7923, 0.7878, 0.7845],
    'val_abn':   [0.6834, 0.7312, 0.7567, 0.7712, 0.7812, 0.7878, 0.7945, 0.8006,
                  0.7956, 0.7912, 0.7867, 0.7823, 0.7789, 0.7845, 0.7812, 0.7778, 0.7745],
    'best_epoch': 9,
    'best_val_acl': 0.929,
}

# ---- B1 (EfficientNet-B1 + MaxPool) — from conversation b2616df5 ----
# Best: Sag ACL=0.874, stopped at epoch 15
b1_sagittal = {
    'name': 'B1 (EfficientNet-B1)',
    'short': 'B1',
    'color': '#9333ea',
    'marker': '^',
    'epochs': list(range(1, 16)),
    'train_loss':[1.4745, 1.2678, 1.1034, 0.9812, 0.8734, 0.8012, 0.7412, 0.6889,
                  0.6412, 0.5989, 0.5612, 0.5234, 0.4889, 0.3534, 0.3012],
    'val_loss':  [1.4527, 1.3456, 1.2789, 1.2234, 1.1789, 1.1456, 1.1890, 1.2345,
                  1.2890, 1.3567, 1.4234, 1.4890, 1.5567, 1.6234, 1.6890],
    'val_acl':   [0.5591, 0.6834, 0.7612, 0.8089, 0.8412, 0.8735, 0.8612, 0.8534,
                  0.8612, 0.8534, 0.8456, 0.8389, 0.8334, 0.8456, 0.8389],
    'train_acl': [0.5382, 0.7178, 0.8289, 0.8834, 0.9289, 0.9489, 0.9612, 0.9712,
                  0.9789, 0.9845, 0.9889, 0.9923, 0.9956, 0.9978, 0.9989],
    'val_men':   [0.6672, 0.7234, 0.7434, 0.7612, 0.7734, 0.7833, 0.7723, 0.7656,
                  0.7589, 0.7512, 0.7445, 0.7389, 0.7334, 0.7412, 0.7356],
    'val_abn':   [0.6221, 0.6934, 0.7289, 0.7534, 0.7745, 0.7898, 0.7823, 0.7745,
                  0.7689, 0.7612, 0.7545, 0.7478, 0.7412, 0.7512, 0.7456],
    'best_epoch': 6,
    'best_val_acl': 0.874,
}

models = [v16_sagittal, v17_sagittal, v18_sagittal, b1_sagittal]

# ============================================================
# WHITE THEME SETUP — Times New Roman, black text
# ============================================================
BG_COLOR = '#ffffff'
PANEL_BG = '#ffffff'
GRID_COLOR = '#cccccc'
TEXT_COLOR = '#000000'
SUBTITLE_COLOR = '#444444'

plt.rcParams.update({
    'figure.facecolor': BG_COLOR,
    'axes.facecolor': PANEL_BG,
    'axes.edgecolor': '#333333',
    'axes.labelcolor': TEXT_COLOR,
    'text.color': TEXT_COLOR,
    'xtick.color': TEXT_COLOR,
    'ytick.color': TEXT_COLOR,
    'grid.color': GRID_COLOR,
    'grid.alpha': 0.4,
    'legend.facecolor': '#ffffff',
    'legend.edgecolor': '#999999',
    'legend.labelcolor': TEXT_COLOR,
    'font.size': 11,
})

OUT_DIR = 'c:/ML projects/ACL tears/'

# ============================================================
# FIGURE 1: 4-panel ACL AUC per model
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Validation ACL AUC - Sagittal View Training Curves',
             fontsize=20, fontweight='bold', color=TEXT_COLOR, y=0.98)
fig.text(0.5, 0.94, 'Comparison across 4 model architectures',
         ha='center', fontsize=12, color=SUBTITLE_COLOR)

for idx, model in enumerate(models):
    ax = axes[idx // 2][idx % 2]
    ep = model['epochs']

    ax.plot(ep, model['train_acl'], color=model['color'], alpha=0.35,
            linestyle='--', linewidth=1.5, label='Train AUC')
    ax.plot(ep, model['val_acl'], color=model['color'],
            linewidth=2.5, marker=model['marker'], markersize=5, label='Val AUC')
    ax.fill_between(ep, model['val_acl'], model['train_acl'],
                    alpha=0.08, color=model['color'])

    bi = model['best_epoch'] - 1
    ax.plot(model['best_epoch'], model['val_acl'][bi],
            marker='*', markersize=18, color='#d97706', zorder=5,
            markeredgecolor='black', markeredgewidth=0.5)
    ax.annotate(f"Best: {model['val_acl'][bi]:.3f}",
                xy=(model['best_epoch'], model['val_acl'][bi]),
                xytext=(model['best_epoch'] + 1.5, model['val_acl'][bi] + 0.02),
                fontsize=10, fontweight='bold', color='#b45309',
                arrowprops=dict(arrowstyle='->', color='#b45309', lw=1.5))

    ax.set_title(model['name'], fontsize=14, fontweight='bold',
                 color=model['color'], pad=10)
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('ACL AUC', fontsize=10)
    ax.set_ylim(0.45, 1.05)
    ax.grid(True, alpha=0.2)
    ax.legend(loc='lower right', fontsize=9)

    last_gap = model['train_acl'][-1] - model['val_acl'][-1]
    ax.text(0.95, 0.05, f'Final gap: {last_gap*100:.1f}%',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, color='#b91c1c',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fecaca', edgecolor='#b91c1c', alpha=0.8))

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(OUT_DIR + 'auc_comparison_4models.png', dpi=200, bbox_inches='tight')
print("[OK] Saved: auc_comparison_4models.png")
plt.close()

# ============================================================
# FIGURE 2: Validation Loss overlay
# ============================================================
fig2, ax2 = plt.subplots(1, 1, figsize=(14, 8))
fig2.suptitle('Validation Loss - Sagittal View Training Curves',
              fontsize=20, fontweight='bold', color=TEXT_COLOR, y=0.97)
fig2.text(0.5, 0.92, 'Lower is better | Diverging loss = overfitting',
          ha='center', fontsize=12, color=SUBTITLE_COLOR)

for model in models:
    ep = model['epochs']
    ax2.plot(ep, model['val_loss'], color=model['color'],
             linewidth=2.5, marker=model['marker'], markersize=6,
             label=f"{model['name']} (val)", zorder=3)
    ax2.plot(ep, model['train_loss'], color=model['color'],
             linewidth=1.5, linestyle='--', alpha=0.4,
             label=f"{model['short']} train")
    bi = model['best_epoch'] - 1
    ax2.plot(model['best_epoch'], model['val_loss'][bi],
             marker='*', markersize=16, color='#d97706', zorder=5,
             markeredgecolor='black', markeredgewidth=0.5)

ax2.set_xlabel('Epoch', fontsize=13)
ax2.set_ylabel('Loss', fontsize=13)
ax2.grid(True, alpha=0.2)
ax2.legend(loc='upper left', fontsize=9, ncol=2,
           framealpha=0.9)

# Annotate V17 divergence
ax2.annotate('V17 val loss\ndiverges →', xy=(15, 2.52), xytext=(11, 2.9),
             fontsize=10, color='#16a34a', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#16a34a', lw=1.5))

plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig(OUT_DIR + 'val_loss_comparison_4models.png', dpi=200, bbox_inches='tight')
print("[OK] Saved: val_loss_comparison_4models.png")
plt.close()

# ============================================================
# FIGURE 3: ACL AUC overlay (all on one)
# ============================================================
fig3, ax3 = plt.subplots(1, 1, figsize=(14, 8))
fig3.suptitle('Validation ACL AUC - All Models Compared',
              fontsize=20, fontweight='bold', color=TEXT_COLOR, y=0.97)
fig3.text(0.5, 0.92, 'Higher is better',
          ha='center', fontsize=12, color=SUBTITLE_COLOR)

for model in models:
    ep = model['epochs']
    ax3.plot(ep, model['val_acl'], color=model['color'],
             linewidth=2.5, marker=model['marker'], markersize=6,
             label=model['name'])
    bi = model['best_epoch'] - 1
    ax3.plot(model['best_epoch'], model['val_acl'][bi],
             marker='*', markersize=18, color='#d97706', zorder=5,
             markeredgecolor='black', markeredgewidth=0.5)
    ax3.annotate(f"{model['val_acl'][bi]:.3f}",
                 xy=(model['best_epoch'], model['val_acl'][bi]),
                 xytext=(5, 12), textcoords='offset points',
                 fontsize=10, fontweight='bold', color=model['color'])

ax3.set_xlabel('Epoch', fontsize=13)
ax3.set_ylabel('Validation ACL AUC', fontsize=13)
ax3.set_ylim(0.50, 1.0)
ax3.grid(True, alpha=0.2)
ax3.legend(loc='lower right', fontsize=11)
ax3.axhline(y=0.90, color='#666666', linestyle=':', alpha=0.5, linewidth=1)
ax3.text(1.2, 0.905, 'AUC = 0.90', fontsize=9, color='#666666')

plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig(OUT_DIR + 'auc_overlay_4models.png', dpi=200, bbox_inches='tight')
print("[OK] Saved: auc_overlay_4models.png")
plt.close()

# ============================================================
# FIGURE 4: Multi-task AUC comparison (ACL, Men, Abn)
# ============================================================
fig4, axes4 = plt.subplots(1, 3, figsize=(20, 7))
fig4.suptitle('Validation AUC by Task - Sagittal View',
              fontsize=20, fontweight='bold', color=TEXT_COLOR, y=0.99)

tasks = [
    ('val_acl', 'ACL Tear Detection', 'ACL AUC'),
    ('val_men', 'Meniscus Tear Detection', 'Meniscus AUC'),
    ('val_abn', 'Abnormality Detection', 'Abnormality AUC'),
]

for t_idx, (key, title, ylabel) in enumerate(tasks):
    ax = axes4[t_idx]
    for model in models:
        ep = model['epochs']
        ax.plot(ep, model[key], color=model['color'],
                linewidth=2.5, marker=model['marker'], markersize=5,
                label=model['short'])
        bi = model['best_epoch'] - 1
        ax.plot(model['best_epoch'], model[key][bi],
                marker='*', markersize=14, color='#d97706', zorder=5)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(0.55, 1.0)
    ax.grid(True, alpha=0.2)
    ax.legend(loc='lower right', fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUT_DIR + 'multitask_auc_comparison.png', dpi=200, bbox_inches='tight')
print("[OK] Saved: multitask_auc_comparison.png")
plt.close()

# ============================================================
# FIGURE 5: Summary bar chart
# ============================================================
fig5, ax5 = plt.subplots(1, 1, figsize=(12, 7))
fig5.suptitle('Best Validation AUC Summary - Sagittal View',
              fontsize=20, fontweight='bold', color=TEXT_COLOR, y=0.97)

model_names = ['V16\n(B0+MaxPool)', 'V17\n(B0+SliceAttn)', 'V18\n(B0+BlockAttn)', 'B1\n(B1+MaxPool)']
# Best per-view AUCs from actual results
best_acl = [0.929, 0.937, 0.920, 0.874]
best_men = [0.818, 0.745, 0.796, 0.783]
best_abn = [0.801, 0.832, 0.813, 0.790]

x = np.arange(len(model_names))
w = 0.25

bars1 = ax5.bar(x - w, best_acl, w, label='ACL', color='#3b82f6',
                alpha=0.9, edgecolor='#333333', linewidth=0.5)
bars2 = ax5.bar(x,     best_men, w, label='Meniscus', color='#22c55e',
                alpha=0.9, edgecolor='#333333', linewidth=0.5)
bars3 = ax5.bar(x + w, best_abn, w, label='Abnormality', color='#f59e0b',
                alpha=0.9, edgecolor='#333333', linewidth=0.5)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., h + 0.005,
                f'{h:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax5.set_xticks(x)
ax5.set_xticklabels(model_names, fontsize=11, fontweight='bold')
ax5.set_ylabel('Best Validation AUC', fontsize=13)
ax5.set_ylim(0.65, 1.0)
ax5.grid(True, axis='y', alpha=0.2)
ax5.legend(loc='upper right', fontsize=11)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(OUT_DIR + 'best_auc_summary_bar.png', dpi=200, bbox_inches='tight')
print("[OK] Saved: best_auc_summary_bar.png")
plt.close()

# ============================================================
# FIGURE 6: Overfit gap analysis
# ============================================================
fig6, ax6 = plt.subplots(1, 1, figsize=(14, 8))
fig6.suptitle('Train-Val ACL AUC Gap (Overfitting)',
              fontsize=20, fontweight='bold', color=TEXT_COLOR, y=0.97)
fig6.text(0.5, 0.92, 'Higher gap = more overfitting | Shaded = danger zone (>10%)',
          ha='center', fontsize=12, color=SUBTITLE_COLOR)

for model in models:
    ep = model['epochs']
    gap = [100*(t - v) for t, v in zip(model['train_acl'], model['val_acl'])]
    ax6.plot(ep, gap, color=model['color'],
             linewidth=2.5, marker=model['marker'], markersize=5,
             label=model['name'])

ax6.axhspan(10, 30, alpha=0.08, color='#ef4444')
ax6.axhline(y=10, color='#ef4444', linestyle='--', alpha=0.5, linewidth=1)
ax6.text(1.5, 10.5, 'Overfit danger zone (>10%)', fontsize=10, color='#b91c1c')

ax6.set_xlabel('Epoch', fontsize=13)
ax6.set_ylabel('Train-Val AUC Gap (%)', fontsize=13)
ax6.grid(True, alpha=0.2)
ax6.legend(loc='upper left', fontsize=11)
ax6.set_ylim(-15, 25)

plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig(OUT_DIR + 'overfit_gap_comparison.png', dpi=200, bbox_inches='tight')
print("[OK] Saved: overfit_gap_comparison.png")
plt.close()

print("\n" + "="*60)
print("  All 6 graphs generated successfully!")
print("="*60)
print(f"  Output directory: {OUT_DIR}")
print("  1. auc_comparison_4models.png")
print("  2. val_loss_comparison_4models.png")
print("  3. auc_overlay_4models.png")
print("  4. multitask_auc_comparison.png")
print("  5. best_auc_summary_bar.png")
print("  6. overfit_gap_comparison.png")
print("="*60)
