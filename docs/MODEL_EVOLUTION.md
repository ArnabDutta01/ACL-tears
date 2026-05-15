# Model Evolution — From 3D CNN to Multi-Task MRNet

## Overview

This project evolved through **18 experimental versions** to determine the optimal
architecture for multi-task knee ligament tear detection from MRI scans. The final
comparison evaluates **4 model configurations** on the Stanford MRNet dataset.

## Evolution Timeline

```
V1 (3D CNN)
 ↓ 3D too expensive, shifted to 2D slices
V3–V4 (2D CNN + Transfer Learning)
 ↓ Adopted MRNet-style per-view architecture
V6–V9 (MRNet + EfficientNet-B0)
 ↓ Added multi-task heads (ACL + Meniscus + Abnormality)
V10–V15 (Multi-task optimization, attention experiments)
 ↓ Converged on final comparison
V16–V18 + B1 (Final architecture comparison)
```

## Key Architectural Decisions

### 1. Per-View Models (MRNet-style)

Following [Bien et al. 2018](https://stanfordmlgroup.github.io/projects/mrnet/),
we train **separate models per MRI view** (sagittal, coronal, axial) rather than
a single model processing all views. Predictions are combined via logistic regression.

**Rationale:** Each view provides distinct anatomical information. Separate models
allow view-specific feature learning without the complexity of 3D or multi-input
architectures.

### 2. EfficientNet-B0 vs B1 Backbone

| Backbone | Parameters | Best ACL AUC | Overfitting |
|----------|-----------|--------------|-------------|
| **EfficientNet-B0** | ~4.0M | **0.929** | Low (5–6%) |
| **EfficientNet-B1** | ~6.5M | 0.874 | High (>10%) |

**Conclusion:** B0's smaller capacity is better suited to the dataset size (~1,130 patients).
B1 overfits despite regularization, suggesting the additional parameters learn noise
rather than signal.

### 3. Max-Pool vs Attention Aggregation

The core question: How to aggregate slice-level features into a volume-level prediction?

| Method | Mechanism | Best ACL AUC | Stability |
|--------|-----------|--------------|-----------|
| **Max-Pool** (V16) | `torch.max(pooled, dim=0)` | **0.929** | ⭐ Most stable |
| Slice Attention (V17) | Learned per-slice weights | 0.937 | Unstable, val loss diverges |
| Block Attention (V18) | Gated multi-head attention | 0.920 | Moderate instability |

**Conclusion:** While V17's slice attention achieves the highest *peak* AUC (0.937),
it is unreliable — validation loss diverges after epoch 7, indicating severe overfitting.
**V16's max-pool is the recommended production model** due to its stability, simplicity,
and strong performance across all three tasks.

### 4. Multi-Task Learning

Training a single model with 3 prediction heads (ACL, Meniscus, Abnormality) outperforms
single-task models because:

- Shared feature extraction captures general musculoskeletal pathology
- Implicit regularization from multiple objectives
- 3× fewer models to maintain (3 views × 1 model vs 3 views × 3 tasks)

## Final Results — V16 (Recommended Model)

### Per-View AUC

| View | ACL | Meniscus | Abnormality |
|------|-----|----------|-------------|
| Sagittal | 0.929 | 0.818 | 0.801 |
| Coronal | 0.896 | 0.793 | 0.842 |
| Axial | 0.872 | 0.761 | 0.789 |
| **Combined** | **0.941** | **0.832** | **0.856** |

### Cross-Model Comparison

| Model | Architecture | ACL AUC | Meniscus AUC | Abnormality AUC | Overfit Gap |
|-------|-------------|---------|--------------|-----------------|-------------|
| **V16** | B0 + MaxPool | 0.929 | 0.818 | 0.801 | **5–6%** |
| V17 | B0 + Slice Attn | **0.937** | 0.745 | 0.832 | 10–15% |
| V18 | B0 + Block Attn | 0.920 | 0.796 | 0.813 | 8–10% |
| B1 | B1 + MaxPool | 0.874 | 0.783 | 0.790 | >10% |

## References

- Bien, N., et al. (2018). "Deep-learning-assisted diagnosis for knee magnetic resonance imaging." *PLOS Medicine*.
- Tan, M., & Le, Q. (2019). "EfficientNet: Rethinking Model Scaling for CNNs." *ICML*.
