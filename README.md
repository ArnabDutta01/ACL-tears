# ACL Tear Detection from Knee MRI Scans

A deep learning project that uses Convolutional Neural Networks (CNNs) to automatically detect Anterior Cruciate Ligament (ACL) tears from knee MRI scans. The project evolved through **three model versions** (V1, V3, V4), progressively improving data sources, preprocessing approaches, and classification strategies.

## 🎯 Project Overview

This project implements a medical imaging AI system that classifies knee MRI scans to detect ACL tears. The system evolved across multiple iterations:

| Version | Approach | Dataset | Input Type | Classification |
|---------|----------|---------|------------|----------------|
| **V1** | 3D CNN | Kaggle (`.pck` files) | 3D MRI volumes (16×128×128) | Binary (Tear vs No Tear) |
| **V3** | 2D CNN | RIMS Hospital (DICOM → `.npz`) | 2D sagittal slices (224×224) | Binary (Tear vs No Tear) |
| **V4** | 2D CNN | Combined (Kaggle + RIMS) | 2D slices (224×224) | Binary + 3-class |

> **Note**: V2 was an intermediate experimental iteration that was not retained.

## 🏗️ Model Versions

### V1 — 3D CNN on Kaggle Data (`V1_3D_kaggle/`)

The original model using 3D convolutions to process volumetric MRI data.

- **Architecture**: Custom 3D CNN with 4 convolutional blocks (32 → 64 → 128 → 256 filters)
- **Input**: 3D MRI volumes resized to 16×128×128, loaded from `.pck` files
- **Dataset**: 917 MRI scans (736 available), sourced from Kaggle
- **Training**: Binary classification (Tear vs No Tear) with BCE loss, Adam optimizer, early stopping
- **Key Files**:
  - `ACL_Tear_Detection_Complete.ipynb` — Main training notebook
  - `acl_detector_model.pth` — Trained model weights (~14 MB)
  - `training_history.png` — Training curves

### V3 — 2D CNN on RIMS Hospital Data (`V3_2D_RIMS/`)

Shifted to 2D approach using real clinical DICOM data from RIMS (Regional Institute of Medical Sciences) hospital.

- **Architecture**: 2D CNN (likely ResNet-based / transfer learning)
- **Input**: 2D sagittal MRI slices (224×224), preprocessed from DICOM format
- **Dataset**: RIMS hospital MRI scans, converted from DICOM to `.npz` format
- **Preprocessing**: Custom pipeline to extract and resize sagittal slices from DICOM series
- **Key Files**:
  - `preprocess_dicom_dataset.py` — DICOM to `.npz` preprocessing pipeline
  - `resize_dataset.py` — Resizing utility for processed slices
  - `ACL_Training_Improved.ipynb` / `ACL_Training_Improved(ran on colab).ipynb` — Training notebook (local + Colab outputs)
  - `ACL_Training_Colab.ipynb` — Colab-specific training notebook

### V4 — 2D CNN on Combined Data (`V4_2D_kaggle+RIMS/`)

Final model combining both Kaggle and RIMS datasets for a larger, more diverse training set. Also explored multi-class classification.

- **Architecture**: 2D CNN for combined dataset
- **Input**: 2D slices (224×224)
- **Dataset**: Combined Kaggle + RIMS data (~1195 samples in the combined set)
- **Classification**: Binary (Tear vs No Tear) + experimental 3-class (No Tear / Partial / Complete)
- **Key Files**:
  - `ACL_Training_Combined.ipynb` / `ACL_Training_Combined(ran on colab).ipynb` — Binary classification training
  - `ACL_Training_Combined_3classes.ipynb` — 3-class classification experiment
  - `PNG_to_Prediction.ipynb` — Inference notebook for making predictions from PNG images
  - `best_acl_model_combined.pth` — Best trained model (~17 MB)

### Grad-CAM Visualizations (`Gradcam/`)

Interpretability analysis using Gradient-weighted Class Activation Mapping to visualize which regions of the MRI the model focuses on for its predictions.

- **Key Files**:
  - `ACL_GradCAM_Visualization.ipynb` — Visualization notebook
  - `gradcam_outputs/` — Generated heatmap overlays showing correct and incorrect predictions

## 📊 Datasets

### Kaggle Dataset (V1)
- **Format**: 3D MRI volumes as `.pck` (pickle) files
- **Organization**: `DATASET/MRI/vol01/` through `vol08/`
- **Metadata**: `metadata.csv` with labels and ROI coordinates
- **Classes**: 0 (No Tear), 1 (Partial Tear), 2 (Complete Tear)
- **Stats**: 917 total samples, 736 available files

### RIMS Hospital Dataset (V3)
- **Source**: Clinical DICOM MRI scans from RIMS hospital
- **Preprocessing**: DICOM → sagittal slice extraction → `.npz` compressed arrays
- **Processed data**: `DATASET/processed_sagittal/` and `DATASET/processed_sagittal_resized/`

### Combined Dataset (V4)
- **Location**: `DATASET/combined/`
- **Format**: `.npz` compressed NumPy arrays with 2D slices
- **Size**: ~1195 `.npz` files + ~466 `.npy` files
- **Metadata**: Separate `metadata.csv` within the combined directory
- **Naming convention**: Files named as `{ID}_{DIAGNOSIS}.npz` (e.g., `001_ACL.npz`, `MRI_329637_NORMAL.npz`)

**⚠️ Note**: The datasets are **NOT** included in this repository due to size and privacy concerns. You'll need to provide your own MRI data.

## 📁 Project Structure

```
ACL-tears/
├── .gitignore                                # Git ignore rules
├── .gitattributes                            # Git LFS & line ending config
├── LICENSE                                   # MIT License
├── README.md                                 # This file
├── PROJECT_DOCUMENTATION.md                  # Detailed technical documentation
│
├── V1_3D_kaggle/                             # Version 1: 3D CNN on Kaggle data
│   ├── ACL_Tear_Detection_Complete.ipynb     #   Main training notebook
│   ├── acl_detector_model.pth               #   [GIT LFS] Trained model (~14 MB)
│   └── training_history.png                  #   Training curves visualization
│
├── V3_2D_RIMS/                               # Version 3: 2D CNN on RIMS data
│   ├── preprocess_dicom_dataset.py           #   DICOM preprocessing pipeline
│   ├── resize_dataset.py                     #   Slice resizing utility
│   ├── ACL_Training_Colab.ipynb              #   Colab training notebook
│   ├── ACL_Training_Improved.ipynb           #   Improved training (local)
│   └── ACL_Training_Improved(ran on colab).ipynb  # Improved training (Colab output)
│
├── V4_2D_kaggle+RIMS/                        # Version 4: 2D CNN on combined data
│   ├── ACL_Training_Combined.ipynb           #   Combined training (local)
│   ├── ACL_Training_Combined(ran on colab).ipynb  # Combined training (Colab output)
│   ├── ACL_Training_Combined_3classes.ipynb  #   3-class classification experiment
│   ├── PNG_to_Prediction.ipynb               #   Inference from PNG images
│   └── best_acl_model_combined.pth           #   [GIT LFS] Best model (~17 MB)
│
├── Gradcam/                                  # GradCAM interpretability
│   ├── ACL_GradCAM_Visualization.ipynb       #   Visualization notebook
│   └── gradcam_outputs/                      #   Heatmap overlay images
│
├── DATASET/                                  # [NOT IN REPO] MRI data
│   ├── MRI/                                  #   Original Kaggle data (vol01–vol08)
│   ├── processed_sagittal/                   #   Processed RIMS sagittal slices
│   ├── processed_sagittal_resized/           #   Resized sagittal slices
│   └── combined/                             #   Combined dataset (.npz/.npy)
│
└── mri_env/                                  # [NOT IN REPO] Python virtual environment
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM
- Google Colab account (optional, for cloud training)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ArnabDutta01/ACL-tears.git
cd ACL-tears
```

2. Create a virtual environment:
```bash
python -m venv mri_env
source mri_env/bin/activate  # On Windows: mri_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install torch torchvision numpy pandas scikit-learn scikit-image matplotlib seaborn pydicom
```

### Running the Models

Each version has its own directory with Jupyter notebooks. Open the desired notebook:

```bash
# V1: 3D CNN
jupyter notebook V1_3D_kaggle/ACL_Tear_Detection_Complete.ipynb

# V3: 2D CNN (RIMS)
jupyter notebook V3_2D_RIMS/ACL_Training_Improved.ipynb

# V4: Combined model
jupyter notebook V4_2D_kaggle+RIMS/ACL_Training_Combined.ipynb
```

Or use Google Colab with the `(ran on colab)` notebook variants.

## 🛠️ Technologies Used

- **PyTorch** — Deep learning framework
- **NumPy** — Numerical computing
- **Pandas** — Data manipulation
- **scikit-image** — Image preprocessing & resizing
- **scikit-learn** — Model evaluation metrics
- **Matplotlib / Seaborn** — Visualization
- **pydicom** — DICOM file parsing (V3)
- **Jupyter / Google Colab** — Interactive development
- **Git LFS** — Large file storage for model weights

## ⚠️ Important Notes

1. **Medical Disclaimer**: This is a research/educational project. Do **NOT** use for actual medical diagnosis without proper clinical validation and regulatory approval.

2. **Data Privacy**: Never commit actual MRI data to version control. Ensure compliance with HIPAA and medical data regulations.

3. **Model Files**: Trained models (`.pth`) are tracked via Git LFS due to their size. Ensure Git LFS is installed before cloning.

4. **Version Numbering**: V2 was an intermediate experiment that was not preserved. The project jumps from V1 directly to V3.

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

## 👥 Author

**ArnabDutta01**

## 🙏 Acknowledgments

- Kaggle MRI dataset contributors
- RIMS (Rajendra Institute of Medical Sciences) for clinical MRI data
- Open-source community for PyTorch, scikit-learn, and related libraries

---

**Note**: This project is for educational and research purposes only.
