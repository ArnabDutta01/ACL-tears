# 📦 Archive — Experimental Version History

This directory contains the full experimental iteration history of the project.
These versions are preserved for reference but are **not part of the final pipeline**.

## Version Catalog

### Phase 1: Initial Exploration (V1–V4)

| Version | Approach | Key Learning |
|---------|----------|--------------|
| **V1** | 3D CNN on Kaggle `.pck` data | 3D convolutions work but are computationally expensive |
| **V3** | 2D CNN on RIMS Hospital DICOM data | 2D slice-based approach is faster and more practical |
| **V4** | 2D CNN on combined Kaggle + RIMS data | Larger datasets improve generalization; explored 3-class |

> V2 and V5 were intermediate experiments that were not preserved as separate versions.

### Phase 2: MRNet Architecture Adoption (V6–V9)

| Version | Approach | Key Learning |
|---------|----------|--------------|
| **V6** | Anti-overfitting with EfficientNet-B0 | Dropout + weight decay reduce train-val gap |
| **V7** | External validation on MRI-only data | Need standardized benchmark for fair comparison |
| **V8** | Multiple Instance Learning (MIL) variants | MIL pooling underperforms simple max-pool on this data |
| **V9** | MRNet-style single-task ACL detection | MRNet architecture with max-pool achieves strong baseline |

### Phase 3: Multi-Task Optimization (V10–V15)

| Version | Approach | Key Learning |
|---------|----------|--------------|
| **V10** | Multi-task (ACL + Meniscus + Abnormal) with 3-class | Multi-task learning improves feature representations |
| **V11** | Composite AUC model selection | Weighted composite metric better than single-task selection |
| **V12** | Enhanced anti-overfitting techniques | Aggressive regularization stabilizes but can hurt peak AUC |
| **V13** | Task-specific attention mechanisms | Attention adds complexity; mixed results |
| **V14** | Architecture refinement | Diminishing returns from added complexity |
| **V15** | Hyperparameter tuning | Confirms that simpler architecture with good tuning is optimal |

### Phase 4: Final Comparison (V16–V18) ← **See main project**

The final 3 model versions (V16, V17, V18) and the B1 backbone comparison
are documented in the main project directory. Original notebooks are preserved
in `old_versions/V16/`, `old_versions/V17/`, and `old_versions/V18/`.

## Old Notebooks

The `old_notebooks/` directory contains notebooks that were used in earlier
phases but are no longer part of the active pipeline:

- `PNG_to_Prediction.ipynb` — Early inference demo (V4 era)
- `slice_visualizer*.ipynb` — MRI slice visualization tools
- `Gradcam/` — Early GradCAM experiments (pre-V16)
- `Notebook/` — Data exploration notebooks
