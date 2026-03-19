# ACL Tear Detection Project - Complete Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Medical Background](#medical-background)
3. [Model Versions](#model-versions)
4. [Technical Architecture](#technical-architecture)
5. [Data Structure and Format](#data-structure-and-format)
6. [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
7. [Deep Learning Model Architecture](#deep-learning-model-architecture)
8. [Training Process](#training-process)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Technologies and Libraries](#technologies-and-libraries)
11. [File Structure](#file-structure)
12. [Key Concepts Explained](#key-concepts-explained)

---

## 1. Project Overview

### What This Project Does
This is a **medical imaging AI system** that uses deep learning to automatically detect Anterior Cruciate Ligament (ACL) tears from knee MRI scans. The project evolved through **three model versions** (V1, V3, V4), each improving upon the previous approach with different data sources, preprocessing techniques, and model architectures.

The system can classify knee MRI images into:
- **Class 0**: No ACL tear (healthy knee)
- **Class 1**: Partial ACL tear (ligament is damaged but not completely torn)
- **Class 2**: Complete ACL tear (ligament is completely ruptured)

> **Note**: V2 was an intermediate experimental iteration that was not retained. The project progresses from V1 directly to V3.

### Primary Goal
The goal is to assist radiologists and medical professionals by providing an automated preliminary diagnosis tool that can:
- Speed up the diagnostic process
- Reduce human error
- Handle large volumes of MRI scans efficiently
- Provide confidence scores for predictions

### Project Type
This is a **supervised learning** problem, specifically:
- **Binary Classification** (V1, V3, V4): Tear vs No Tear
- **Multi-class Classification** (V4 experimental): No Tear vs Partial Tear vs Complete Tear
- **Computer Vision** task using **3D Medical Imaging** (V1) and **2D Medical Imaging** (V3, V4)

---

## 2. Medical Background

### What is the ACL?
The **Anterior Cruciate Ligament (ACL)** is one of the four major ligaments in the knee that connects the femur (thighbone) to the tibia (shinbone). It provides:
- Rotational stability to the knee
- Prevention of the tibia from sliding forward relative to the femur
- Critical support during pivoting and cutting movements

### Why ACL Tears Matter
- **Common Sports Injury**: Affects approximately 200,000 people annually in the US
- **Severe Impact**: Can sideline athletes for 6-12 months
- **Requires Accurate Diagnosis**: Proper diagnosis is critical for treatment planning
- **Treatment Options**: May require surgical reconstruction or physical therapy

### How MRI Helps
**MRI (Magnetic Resonance Imaging)** is the gold standard for ACL tear diagnosis because:
- **Non-invasive**: No surgery or radiation needed
- **High Soft Tissue Contrast**: Excellent for visualizing ligaments, tendons, and cartilage
- **3D Visualization**: Provides volumetric data showing the knee from multiple angles
- **Diagnostic Accuracy**: Can differentiate between partial and complete tears

### Traditional Diagnosis Process
1. Patient presents with knee pain/instability
2. Physical examination (Lachman test, anterior drawer test)
3. MRI scan ordered for confirmation
4. Radiologist manually reviews hundreds of MRI slices
5. Diagnosis made based on ligament appearance (continuity, signal intensity, orientation)

### How AI Improves This
- **Automation**: Reduces manual review time
- **Consistency**: Eliminates inter-observer variability
- **Speed**: Can process scans in seconds vs minutes/hours
- **Assistance**: Serves as a "second opinion" for radiologists

---

## 3. Model Versions

### Version Evolution Summary

```
V1 (3D CNN, Kaggle)
    ↓ Learned: 3D approach works but limited by single data source
V3 (2D CNN, RIMS Hospital)
    ↓ Shifted to 2D approach with real clinical DICOM data
V4 (2D CNN, Kaggle + RIMS Combined)
    ↓ Combined datasets for larger, more diverse training set
    ↓ Explored multi-class (3-class) classification
```

### V1 — 3D CNN on Kaggle Data

**Location**: `V1_3D_kaggle/`

The original model using 3D convolutions to process volumetric MRI data directly.

| Aspect | Details |
|--------|---------|
| **Input** | 3D MRI volumes (16×128×128) from `.pck` files |
| **Architecture** | Custom 3D CNN: 4 conv blocks (32→64→128→256 filters) |
| **Classification** | Binary (Tear vs No Tear) |
| **Dataset** | Kaggle: 917 scans (736 available) |
| **Loss** | BCELoss with optional class weighting |
| **Optimizer** | Adam (lr=0.001) with ReduceLROnPlateau |
| **Key Output** | `acl_detector_model.pth` (~14 MB) |

**Strengths**: Preserves 3D spatial context between MRI slices.
**Limitations**: Computationally expensive; single data source; smaller effective dataset.

### V3 — 2D CNN on RIMS Hospital Data

**Location**: `V3_2D_RIMS/`

Shifted to a 2D approach using real clinical data sourced from RIMS (Regional Institute of Medical Sciences) hospital. Includes custom preprocessing scripts to convert DICOM files to model-ready formats.

| Aspect | Details |
|--------|---------|
| **Input** | 2D sagittal MRI slices (224×224) from `.npz` files |
| **Architecture** | 2D CNN (improved architecture / transfer learning) |
| **Classification** | Binary (Tear vs No Tear) |
| **Dataset** | RIMS hospital clinical DICOM scans |
| **Preprocessing** | DICOM → sagittal extraction → resizing → `.npz` |
| **Training** | Local + Google Colab |

**Key Preprocessing Scripts**:
- `preprocess_dicom_dataset.py` — Full DICOM to `.npz` pipeline
- `resize_dataset.py` — Standardizes slice dimensions to 224×224

**Strengths**: Uses real clinical data; 2D approach is faster and more memory-efficient.
**Limitations**: Smaller dataset from a single hospital.

### V4 — 2D CNN on Combined Data

**Location**: `V4_2D_kaggle+RIMS/`

Final iteration combining both datasets for a larger, more diverse training set. Also included an experimental 3-class classification attempt.

| Aspect | Details |
|--------|---------|
| **Input** | 2D slices (224×224) from combined `.npz` dataset |
| **Architecture** | 2D CNN for combined dataset |
| **Classification** | Binary (primary) + 3-class (experimental) |
| **Dataset** | Combined Kaggle + RIMS (~1195 samples) |
| **Key Output** | `best_acl_model_combined.pth` (~17 MB) |

**Notebooks**:
- `ACL_Training_Combined.ipynb` — Binary classification
- `ACL_Training_Combined_3classes.ipynb` — 3-class experiment (No Tear / Partial / Complete)
- `PNG_to_Prediction.ipynb` — Inference pipeline for PNG images

**Strengths**: Largest and most diverse dataset; explored multi-class classification.

### Grad-CAM Interpretability

**Location**: `Gradcam/`

Uses Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize which regions of the MRI the model focuses on when making predictions — crucial for medical AI trust and validation.

- `ACL_GradCAM_Visualization.ipynb` — Generates heatmap overlays
- `gradcam_outputs/` — Contains sample correct and incorrect prediction visualizations

---

## 4. Technical Architecture

### Overall System Design

**V1 (3D Approach)**:
```
Input: 3D MRI Volume (.pck file)
    ↓
ROI Extraction + Normalization + Resize (16×128×128)
    ↓
3D Convolutional Neural Network
    ↓
Binary Classification
    ↓
Output: Tear (1) or No Tear (0) + Confidence Score
```

**V3/V4 (2D Approach)**:
```
Input: DICOM / MRI Volume
    ↓
Sagittal Slice Extraction + Resize (224×224)
    ↓
2D Convolutional Neural Network
    ↓
Binary / Multi-class Classification
    ↓
Output: Prediction + Confidence Score
```

### Why 3D CNN (V1) vs 2D CNN (V3/V4)?

**3D Convolution (V1)**:
- Processes volume as a whole → captures spatial relationships between slices
- Higher computational cost and memory usage
- Requires consistent 3D volume input

**2D Convolution (V3/V4)**:
- Processes individual sagittal slices → faster training
- Can leverage pre-trained 2D models (transfer learning)
- Loses some inter-slice context but gains efficiency
- More practical for clinical deployment

---

## 5. Data Structure and Format

### Kaggle Dataset (V1)

```
DATASET/MRI/
├── metadata.csv           # Labels and ROI coordinates
├── vol01/                 # Volume folder 1
│   ├── 0000.pck          # MRI scan files
│   ├── 0001.pck
│   └── ...
├── vol02/ ... vol08/      # Volume folders 2-8
```

#### Metadata CSV Structure

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `examId` | Integer | Unique identifier for each MRI exam | 1234 |
| `seriesNo` | Integer | Series number within the exam | 5 |
| `aclDiagnosis` | Integer | **Target label**: 0=No tear, 1=Partial, 2=Complete | 1 |
| `kneeLR` | Integer | Which knee: 0=Left, 1=Right | 0 |
| `roiX` | Integer | X-coordinate of ROI starting point | 50 |
| `roiY` | Integer | Y-coordinate of ROI starting point | 120 |
| `roiZ` | Integer | Z-coordinate (slice) of ROI starting point | 10 |
| `roiWidth` | Integer | Width of the ROI in pixels | 64 |
| `roiHeight` | Integer | Height of the ROI in pixels | 128 |
| `roiDepth` | Integer | Depth of the ROI (number of slices) | 16 |
| `volumeFilename` | String | Name of the .pck file containing MRI data | 0000.pck |

### RIMS Hospital Dataset (V3)

Processed from DICOM files using `preprocess_dicom_dataset.py`:

```
DATASET/
├── processed_sagittal/              # Extracted sagittal slices
│   ├── 001_ACL.npz
│   ├── 002_PARTIAL_ACL.npz
│   └── ...
├── processed_sagittal_resized/      # Resized to 224×224
│   ├── 001_ACL.npz
│   └── ...
```

### Combined Dataset (V4)

```
DATASET/combined/
├── metadata.csv           # Labels for all samples
├── 001_ACL.npz           # Compressed 2D slices
├── 002_PARTIAL_ACL.npz
├── ...
├── MRI_329637_NORMAL.npz  # Kaggle-sourced samples
├── MRI_390116_NORMAL.npz
└── ...
```

**Naming Convention**:
- `{ID}_{DIAGNOSIS}.npz` for RIMS data (e.g., `001_ACL.npz`)
- `MRI_{ID}_{DIAGNOSIS}.npz` for Kaggle data (e.g., `MRI_329637_NORMAL.npz`)

### What is ROI (Region of Interest)?

**ROI** = The specific 3D region in the MRI scan where the ACL is located (used in V1)

**Why ROI is Important**:
- MRI scans contain the entire knee, but we only need the ACL region
- Focusing on ROI reduces computation and noise
- ROI is manually annotated by medical experts
- ROI coordinates define a 3D bounding box: `[z:z+depth, y:y+height, x:x+width]`

**Visual Representation**:
```
Full MRI Volume: [40 slices, 256×256 pixels]
                      ↓
Extract ROI: [z=10:26, y=120:248, x=50:114]
                      ↓
ROI Volume: [16 slices, 128×64 pixels]
```

### MRI File Formats

#### .pck files (V1 — Kaggle)
Python pickle files containing 3D NumPy arrays:
```python
with open('0000.pck', 'rb') as f:
    mri_volume = pickle.load(f)
# Shape: (40, 256, 256) — 40 slices, each 256×256 pixels
```

#### .npz files (V3/V4 — Processed)
Compressed NumPy archives containing 2D sagittal slices:
```python
data = np.load('001_ACL.npz')
slices = data['arr_0']  # 2D array of sagittal slices
```

### Dataset Statistics

**V1 (Kaggle)**:
- **Total Samples**: 917 MRI scans in metadata
- **Available Files**: 736 scans
- **Class Distribution**: No Tear 690 (75%) | Tear 227 (25%)

**V4 (Combined)**:
- **Total Samples**: ~1195 combined entries
- **Sources**: Kaggle + RIMS hospital

---

## 6. Data Preprocessing Pipeline

### V1 — 3D Pipeline (Kaggle Data)

#### Step 1: File Location
```python
def find_pck_file(data_dir, filename):
    for vol_num in range(1, 9):
        vol_folder = f"vol{vol_num:02d}"
        path = os.path.join(data_dir, vol_folder, filename)
        if os.path.exists(path):
            return path
```

#### Step 2: Load MRI Volume
```python
with open(file_path, 'rb') as f:
    mri_volume = pickle.load(f)
```

#### Step 3: Extract ROI
```python
roi = mri_volume[z:z+depth, y:y+height, x:x+width]
```

#### Step 4: Normalization
```python
roi = (roi - roi.min()) / (roi.max() - roi.min() + 1e-8)
```
Scales all pixel values to `[0, 1]`. The epsilon `(1e-8)` prevents division by zero.

#### Step 5: Resizing
```python
from skimage.transform import resize
roi = resize(roi, (16, 128, 128), mode='constant', anti_aliasing=True)
```
Standardizes all ROI volumes to the same dimensions.

#### Step 6: Convert to PyTorch Tensor
```python
x = torch.tensor(roi).unsqueeze(0)
# Shape: (16, 128, 128) → (1, 16, 128, 128) [channels, depth, height, width]
```

### V3 — 2D Pipeline (RIMS DICOM Data)

The `preprocess_dicom_dataset.py` script handles the full pipeline:

1. **Read DICOM series** from hospital scanner output
2. **Extract sagittal plane** slices from the 3D volume
3. **Select relevant slices** (middle sagittal region where ACL is best visualized)
4. **Normalize** intensity values to `[0, 1]`
5. **Save as `.npz`** compressed NumPy arrays

The `resize_dataset.py` script then resizes all slices to a consistent `224×224` resolution.

### V4 — Combined Pipeline

Merges the processed outputs from both Kaggle and RIMS datasets into a unified `DATASET/combined/` directory with a single `metadata.csv` for training.

---

## 7. Deep Learning Model Architecture

### V1 Model: ACLDetector3D

```python
class ACLDetector3D(nn.Module):
    def __init__(self):
        # 4 convolutional blocks + fully connected classifier
```

#### Complete Architecture Breakdown

**Input**: `(batch_size, 1, 16, 128, 128)` — (batch, channels, depth, height, width)

**Convolutional Block 1**:
```python
nn.Conv3d(1, 32, kernel_size=3, padding=1)    # → (batch, 32, 16, 128, 128)
nn.BatchNorm3d(32)
nn.ReLU(inplace=True)
nn.MaxPool3d(kernel_size=2, stride=2)          # → (batch, 32, 8, 64, 64)
```

**Convolutional Block 2**:
```python
nn.Conv3d(32, 64, kernel_size=3, padding=1)    # → (batch, 64, 8, 64, 64)
nn.BatchNorm3d(64)
nn.ReLU(inplace=True)
nn.MaxPool3d(kernel_size=2, stride=2)          # → (batch, 64, 4, 32, 32)
```

**Convolutional Block 3**:
```python
nn.Conv3d(64, 128, kernel_size=3, padding=1)   # → (batch, 128, 4, 32, 32)
nn.BatchNorm3d(128)
nn.ReLU(inplace=True)
nn.MaxPool3d(kernel_size=2, stride=2)          # → (batch, 128, 2, 16, 16)
```

**Convolutional Block 4**:
```python
nn.Conv3d(128, 256, kernel_size=3, padding=1)  # → (batch, 256, 2, 16, 16)
nn.BatchNorm3d(256)
nn.ReLU(inplace=True)
nn.AdaptiveAvgPool3d((1, 1, 1))                # → (batch, 256, 1, 1, 1)
```

**Classifier (Fully Connected)**:
```python
nn.Flatten()                    # (batch, 256)
nn.Linear(256, 128)
nn.ReLU(inplace=True)
nn.Dropout(0.5)
nn.Linear(128, 64)
nn.ReLU(inplace=True)
nn.Dropout(0.3)
nn.Linear(64, 1)
nn.Sigmoid()                    # Output: probability in [0, 1]
```

**Total Parameters**: ~1.25M trainable parameters

#### Model Summary (V1)
```
Input: (batch, 1, 16, 128, 128)
↓ Conv Block 1: (batch, 32, 8, 64, 64)
↓ Conv Block 2: (batch, 64, 4, 32, 32)
↓ Conv Block 3: (batch, 128, 2, 16, 16)
↓ Conv Block 4: (batch, 256, 1, 1, 1)
↓ Flatten: (batch, 256)
↓ FC Layers: (batch, 128) → (batch, 64) → (batch, 1)
Output: (batch, 1) — Probability in [0, 1]
```

### V3/V4 Model Architecture

The V3 and V4 models use 2D CNN architectures, likely leveraging transfer learning with pre-trained backbones (e.g., ResNet). The 2D approach:
- Processes individual 224×224 sagittal slices
- Uses `nn.Conv2d` instead of `nn.Conv3d`
- Benefits from pre-trained ImageNet weights
- Lower memory requirements compared to 3D approach

---

## 8. Training Process

### Data Splitting Strategy

#### Train-Validation-Test Split
```python
train_df, temp_df = train_test_split(valid_data, test_size=0.3, stratify=valid_data['binary_label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['binary_label'])
```

**Proportions**:
- **Training**: 70% — Used to train the model
- **Validation**: 15% — Used during training for hyperparameter tuning
- **Test**: 15% — Final evaluation, never seen during training

**Stratified Splitting**: Ensures each split maintains the same class distribution.

### DataLoader and Batching

```python
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
```

- **Batch Size = 8**: Processes 8 MRI samples at once
- **shuffle=True** (training): Randomly reorders samples each epoch
- **shuffle=False** (validation/test): Consistent ordering for reproducibility

### Loss Function

#### Binary Cross-Entropy (BCE) Loss
```python
criterion = nn.BCELoss()
```

**Formula**: `BCE = -[y * log(ŷ) + (1-y) * log(1-ŷ)]`

#### Weighted BCE (for Imbalanced Data)
```python
pos_weight = torch.tensor([n_negative / n_positive])  # ~3.0
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

### Optimizer

#### Adam Optimizer
```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### Learning Rate Scheduler

```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)
```

Automatically reduces learning rate by 50% when validation loss plateaus for 3 epochs.

### Early Stopping

```python
PATIENCE = 7
if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_model_state = model.state_dict().copy()
    patience_counter = 0
else:
    patience_counter += 1
if patience_counter >= PATIENCE:
    print("Early stopping triggered")
    break
```

Stops training when validation loss stops improving for 7 consecutive epochs.

### Training Configuration (V1)

```python
NUM_EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 0.001
PATIENCE = 7
TARGET_SIZE = (16, 128, 128)  # V1: 3D
```

---

## 9. Evaluation Metrics

### Accuracy
```python
accuracy = correct_predictions / total_predictions
```

**Limitations**: Misleading with imbalanced data (e.g., always predicting "No Tear" gives 75% accuracy).

### Confusion Matrix
```
                Predicted
              No Tear  |  Tear
Actual -------------------------
No Tear   |    TN     |   FP
Tear      |    FN     |   TP
```

**Medical Significance**:
- **False Negative (FN)**: Most dangerous — missed tear, patient doesn't get treatment
- **False Positive (FP)**: Less dangerous but causes unnecessary procedures

### Precision, Recall, F1-Score

- **Precision** = TP / (TP + FP) — "Of predicted tears, how many were actual tears?"
- **Recall** = TP / (TP + FN) — "Of actual tears, how many did we detect?"
- **F1-Score** = 2 × (Precision × Recall) / (Precision + Recall) — Harmonic mean

### Classification Report
```python
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred, target_names=['No Tear', 'ACL Tear']))
```

---

## 10. Technologies and Libraries

| Library | Purpose |
|---------|---------|
| **PyTorch** (`torch`, `torchvision`) | Deep learning framework, model definition, training |
| **NumPy** | Array operations, numerical computing |
| **Pandas** | Data manipulation, CSV loading |
| **scikit-image** (`skimage`) | Image resizing and preprocessing |
| **scikit-learn** (`sklearn`) | Train/test splitting, evaluation metrics |
| **Matplotlib** | Visualization, training curves, confusion matrices |
| **Seaborn** | Statistical visualizations |
| **pydicom** | DICOM medical image file parsing (V3) |
| **Jupyter / Google Colab** | Interactive development environment |
| **Git LFS** | Large file storage for model weights |
| **pickle** | Loading `.pck` MRI files (V1) |

---

## 11. File Structure

```
ACL tears/
├── .git/                                      # Git repository
├── .gitignore                                 # Git ignore rules
├── .gitattributes                             # Git LFS & line ending config
├── LICENSE                                    # MIT License
├── README.md                                  # Project overview
├── PROJECT_DOCUMENTATION.md                   # This file
│
├── V1_3D_kaggle/                              # Version 1: 3D CNN on Kaggle data
│   ├── ACL_Tear_Detection_Complete.ipynb      #   Main training notebook
│   ├── acl_detector_model.pth                 #   Trained model weights (~14 MB)
│   └── training_history.png                   #   Training curves
│
├── V3_2D_RIMS/                                # Version 3: 2D CNN on RIMS data
│   ├── preprocess_dicom_dataset.py            #   DICOM preprocessing pipeline
│   ├── resize_dataset.py                      #   Slice resizing utility
│   ├── ACL_Training_Colab.ipynb               #   Colab training notebook
│   ├── ACL_Training_Improved.ipynb            #   Improved training (local)
│   └── ACL_Training_Improved(ran on colab).ipynb  # With Colab outputs
│
├── V4_2D_kaggle+RIMS/                         # Version 4: 2D CNN on Combined data
│   ├── ACL_Training_Combined.ipynb            #   Binary classification training
│   ├── ACL_Training_Combined(ran on colab).ipynb  # With Colab outputs
│   ├── ACL_Training_Combined_3classes.ipynb   #   3-class classification experiment
│   ├── PNG_to_Prediction.ipynb                #   Inference from PNG images
│   └── best_acl_model_combined.pth            #   Best model weights (~17 MB)
│
├── Gradcam/                                   # GradCAM interpretability
│   ├── ACL_GradCAM_Visualization.ipynb        #   Visualization notebook
│   └── gradcam_outputs/                       #   Heatmap overlay images
│
├── DATASET/                                   # [NOT IN REPO] All MRI data
│   ├── MRI/                                   #   Kaggle data (vol01–vol08 + metadata)
│   │   ├── metadata.csv
│   │   ├── vol01/ ... vol08/
│   ├── processed_sagittal/                    #   Processed RIMS sagittal slices
│   ├── processed_sagittal_resized/            #   Resized sagittal slices
│   └── combined/                              #   Combined dataset + metadata
│       ├── metadata.csv
│       ├── *.npz
│       └── *.npy
│
└── mri_env/                                   # [NOT IN REPO] Virtual environment
```

### Key Files Explained

#### V1: ACL_Tear_Detection_Complete.ipynb
- Main 3D CNN training notebook
- Contains complete pipeline from data loading to evaluation
- Binary classification with `.pck` file processing
- Well-documented with markdown explanations

#### V3: preprocess_dicom_dataset.py
- Converts raw DICOM files from hospital scanners to `.npz` format
- Extracts sagittal plane slices
- Handles various DICOM metadata and orientations

#### V4: ACL_Training_Combined_3classes.ipynb
- Experimental notebook attempting 3-class classification
- Classes: No Tear (0), Partial Tear (1), Complete Tear (2)
- Uses CrossEntropyLoss instead of BCELoss

#### Model Files (.pth)
- `V1_3D_kaggle/acl_detector_model.pth` (~14 MB) — V1 trained weights
- `V4_2D_kaggle+RIMS/best_acl_model_combined.pth` (~17 MB) — V4 best model

Both contain `model_state_dict`, `optimizer_state_dict`, and training history. Tracked via Git LFS.

#### Grad-CAM Outputs
- `gradcam_outputs/` contains heatmap overlays showing correct and incorrect predictions
- File naming: `{ID}_{DIAGNOSIS}_{correct/incorrect}.png`

---

## 12. Key Concepts Explained

### 1. Binary Classification

**In This Project**:
- **Class 0**: No ACL Tear (negative class)
- **Class 1**: ACL Tear (positive class)

**Converting Multi-class to Binary**:
```python
metadata['binary_label'] = (metadata['aclDiagnosis'] > 0).astype(int)
# 0 → 0 (no tear)
# 1 → 1 (partial tear → tear)
# 2 → 1 (complete tear → tear)
```

### 2. 3D vs 2D Convolution

**3D Convolution (V1)**:
- Kernel is a 3D cube (e.g., 3×3×3)
- Slides over volume in all three dimensions
- Captures inter-slice spatial relationships
- Higher computational cost

**2D Convolution (V3/V4)**:
- Kernel is a 2D grid (e.g., 3×3)
- Processes individual slices
- Can leverage pre-trained models
- More computationally efficient

### 3. Class Imbalance

**Problem**: No Tear: 690 (75%) vs Tear: 227 (25%)

**Solutions Used**:
1. Weighted Loss — More weight to minority class
2. Stratified Splitting — Maintain class ratio across splits
3. Evaluation Metrics — Use F1/precision/recall beyond accuracy

### 4. Overfitting vs Underfitting

**Overfitting**: Model memorizes training data; high train accuracy, low validation accuracy.
- Solutions: Dropout, early stopping, data augmentation, more data

**Underfitting**: Model too simple; both train and validation accuracy are low.
- Solutions: Larger model, more training, better features

### 5. Batch Normalization

Normalizes layer inputs to `mean=0, std=1`:
```
x_normalized = (x - μ_batch) / √(σ²_batch + ε)
output = γ * x_normalized + β
```
Benefits: Faster training, higher learning rates, regularization effect.

### 6. Dropout

Randomly sets neurons to zero during training:
```python
nn.Dropout(0.5)  # 50% dropout rate
```
Prevents co-adaptation of neurons and acts as regularization. Disabled during evaluation.

### 7. Transfer Learning (V3/V4)

Using pre-trained models (e.g., ResNet trained on ImageNet) as feature extractors, then fine-tuning the final layers on medical data. This is especially valuable when medical datasets are small.

### 8. DICOM Format (V3)

DICOM (Digital Imaging and Communications in Medicine) is the standard format for medical imaging. The `pydicom` library is used to parse these files, which contain both pixel data and rich metadata (patient info, scan parameters, etc.).

### 9. Grad-CAM

Gradient-weighted Class Activation Mapping generates heatmaps showing which regions of the input image the model focuses on. This is crucial for:
- **Interpretability**: Understanding model decisions
- **Trust**: Validating that the model looks at clinically relevant regions
- **Debugging**: Identifying when the model uses spurious features

### 10. GPU Acceleration

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
data = data.to(device)
```

GPUs provide 10-100× speedup over CPUs for deep learning due to parallel matrix operations.

---

## Conclusion

This ACL tear detection project demonstrates a complete, evolving deep learning pipeline for medical image analysis across three model versions:

1. **V1 (3D CNN)**: Established the baseline using 3D convolutions on Kaggle volumetric data
2. **V3 (2D CNN)**: Shifted to 2D approach with real clinical DICOM data from RIMS hospital
3. **V4 (2D CNN)**: Combined datasets for improved diversity; explored multi-class classification
4. **Grad-CAM**: Added model interpretability through activation visualization

**Key Takeaways**:
- Medical imaging requires specialized preprocessing (ROI extraction, DICOM parsing, normalization)
- Both 3D and 2D CNN approaches have trade-offs for volumetric medical data
- Combining multiple data sources improves model robustness
- Class imbalance must be addressed in medical datasets
- Model interpretability (Grad-CAM) is essential for clinical trust
- Proper evaluation metrics (beyond accuracy) are critical

**Potential Applications**:
- Assist radiologists in preliminary screening
- Reduce diagnosis time
- Second opinion system
- Educational tool for medical students

**Important Disclaimer**: This is a research/educational project and should NOT be used for actual medical diagnosis without proper clinical validation, regulatory approval, and oversight by medical professionals.

---

**End of Documentation**
