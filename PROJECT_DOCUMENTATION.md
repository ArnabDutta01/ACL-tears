# ACL Tear Detection Project - Complete Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Medical Background](#medical-background)
3. [Technical Architecture](#technical-architecture)
4. [Data Structure and Format](#data-structure-and-format)
5. [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
6. [Deep Learning Model Architecture](#deep-learning-model-architecture)
7. [Training Process](#training-process)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Technologies and Libraries](#technologies-and-libraries)
10. [File Structure](#file-structure)
11. [Key Concepts Explained](#key-concepts-explained)

---

## 1. Project Overview

### What This Project Does
This is a **medical imaging AI system** that uses deep learning to automatically detect Anterior Cruciate Ligament (ACL) tears from knee MRI scans. The system can classify knee MRI images into three categories:
- **Class 0**: No ACL tear (healthy knee)
- **Class 1**: Partial ACL tear (ligament is damaged but not completely torn)
- **Class 2**: Complete ACL tear (ligament is completely ruptured)

### Primary Goal
The goal is to assist radiologists and medical professionals by providing an automated preliminary diagnosis tool that can:
- Speed up the diagnostic process
- Reduce human error
- Handle large volumes of MRI scans efficiently
- Provide confidence scores for predictions

### Project Type
This is a **supervised learning** problem, specifically:
- **Binary Classification** (as implemented): Tear vs No Tear
- **Multi-class Classification** (potential): No Tear vs Partial Tear vs Complete Tear
- **Computer Vision** task using **3D Medical Imaging**

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

## 3. Technical Architecture

### Overall System Design

```
Input: 3D MRI Volume (.pck file)
    ↓
Preprocessing Pipeline
    ↓
3D Convolutional Neural Network
    ↓
Binary Classification
    ↓
Output: Tear (1) or No Tear (0) + Confidence Score
```

### Why 3D CNN Instead of 2D CNN?

**3D Convolution** is essential because:
- MRI scans are volumetric (depth × height × width)
- ACL tears may be visible in certain slices but not others
- Spatial relationships between slices contain diagnostic information
- A 2D CNN would lose inter-slice context

**Comparison**:
- **2D CNN**: Processes each slice independently → loses 3D context
- **3D CNN**: Processes volume as a whole → captures spatial relationships

---

## 4. Data Structure and Format

### Dataset Organization

```
DATASET/MRI/
├── metadata.csv           # Labels and ROI coordinates
├── vol01/                 # Volume folder 1
│   ├── 0000.pck          # MRI scan files
│   ├── 0001.pck
│   └── ...
├── vol02/                 # Volume folder 2
├── vol03/
├── vol04/
├── vol05/
├── vol06/
├── vol07/
└── vol08/
```

### Metadata CSV Structure

**metadata.csv** contains the following columns:

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

### What is ROI (Region of Interest)?

**ROI** = The specific 3D region in the MRI scan where the ACL is located

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

### MRI File Format (.pck)

**.pck files** are Python pickle files containing:
- **3D NumPy array**: Shape is typically (num_slices, height, width)
- **Data type**: Usually float32 or uint16
- **Intensity values**: Represent tissue signal intensity from MRI scanner

**Example**:
```python
# Loading a .pck file
with open('0000.pck', 'rb') as f:
    mri_volume = pickle.load(f)

# Shape: (40, 256, 256)
# 40 slices, each 256×256 pixels
```

### Dataset Statistics

From the README:
- **Total Samples**: 917 MRI scans in metadata
- **Available Files**: 736 scans (some files may be missing)
- **Class Distribution**:
  - No Tear (0): 690 samples (75%)
  - Partial Tear (1): ~150 samples
  - Complete Tear (2): ~77 samples
- **After Binary Conversion**:
  - No Tear (0): 690 samples (75%)
  - Any Tear (1): 227 samples (25%)

**Class Imbalance**: This is an imbalanced dataset, which affects training strategy.

---

## 5. Data Preprocessing Pipeline

### Step-by-Step Preprocessing

#### Step 1: File Location
```python
def find_pck_file(data_dir, filename):
    # Searches vol01 through vol08 folders
    for vol_num in range(1, 9):
        vol_folder = f"vol{vol_num:02d}"
        path = os.path.join(data_dir, vol_folder, filename)
        if os.path.exists(path):
            return path
```
**Purpose**: Locate the MRI file across multiple volume folders

#### Step 2: Load MRI Volume
```python
with open(file_path, 'rb') as f:
    mri_volume = pickle.load(f)
# Result: 3D numpy array
```
**Purpose**: Load the pickled NumPy array into memory

#### Step 3: Extract ROI
```python
roi = mri_volume[z:z+depth, y:y+height, x:x+width]
```
**Purpose**: Crop the volume to the ACL region using metadata coordinates

#### Step 4: Normalization
```python
roi = (roi - roi.min()) / (roi.max() - roi.min() + 1e-8)
```
**What This Does**:
- Scales all pixel values to the range [0, 1]
- **Why**: Neural networks perform better with normalized inputs
- **Formula**: `normalized = (value - min) / (max - min)`
- **epsilon (1e-8)**: Prevents division by zero

**Example**:
```
Original: [100, 500, 1000, 2000]
Min: 100, Max: 2000
Normalized: [0.0, 0.21, 0.47, 1.0]
```

#### Step 5: Resizing
```python
from skimage.transform import resize
roi = resize(roi, (16, 128, 128), mode='constant', anti_aliasing=True)
```
**Purpose**: Standardize all ROI volumes to the same size

**Why Resize?**:
- Neural networks require fixed input dimensions
- Different ROIs have different sizes in the original data
- Target size (16, 128, 128) balances detail vs computation

**Parameters Explained**:
- `mode='constant'`: Pad with zeros if needed
- `anti_aliasing=True`: Smooth downsampling to prevent artifacts
- `preserve_range=True`: Keep values in [0, 1] range

**What Happens**:
```
Input ROI:  [24 slices, 142×88 pixels]
           ↓ (interpolation)
Output ROI: [16 slices, 128×128 pixels]
```

#### Step 6: Convert to PyTorch Tensor
```python
x = torch.tensor(roi).unsqueeze(0)
```
**Shape Transformation**:
```
roi shape:      (16, 128, 128)          # depth, height, width
unsqueeze(0):   (1, 16, 128, 128)       # channels, depth, height, width
```
**Why add channel dimension?**: CNN expects input format [batch, channels, depth, height, width]

### Data Augmentation (Not Implemented, But Important)

**Potential Augmentations**:
- **Rotation**: Rotate volume by small angles (±10°)
- **Flipping**: Horizontal/vertical flips
- **Intensity Variation**: Adjust brightness/contrast
- **Elastic Deformation**: Simulate tissue deformation
- **Noise Addition**: Add Gaussian noise to simulate different scanner settings

**Why Not Used Here**:
- Increases training time
- Requires careful implementation for medical images
- Could be added to improve model performance

---

## 6. Deep Learning Model Architecture

### Model Class: ACLDetector3D

```python
class ACLDetector3D(nn.Module):
    def __init__(self):
        # 4 convolutional blocks + fully connected classifier
```

### Complete Architecture Breakdown

#### Input Layer
```
Input shape: (batch_size, 1, 16, 128, 128)
             (batch,     channels, depth, height, width)
```

#### Convolutional Block 1
```python
nn.Conv3d(1, 32, kernel_size=3, padding=1)
nn.BatchNorm3d(32)
nn.ReLU(inplace=True)
nn.MaxPool3d(kernel_size=2, stride=2)
```

**Detailed Explanation**:

**Conv3d(1, 32, kernel_size=3, padding=1)**:
- **Input channels**: 1 (grayscale MRI)
- **Output channels**: 32 (learns 32 different 3D feature patterns)
- **Kernel size**: 3×3×3 cube that slides over the volume
- **Padding**: 1 pixel border added to maintain size
- **Output**: (batch, 32, 16, 128, 128)

**BatchNorm3d(32)**:
- Normalizes the 32 feature maps
- **Why**: Stabilizes training, allows higher learning rates
- **How**: Normalizes each channel to mean=0, std=1

**ReLU (Rectified Linear Unit)**:
- Activation function: `f(x) = max(0, x)`
- **Why**: Introduces non-linearity, allows learning complex patterns
- **inplace=True**: Saves memory by modifying input directly

**MaxPool3d(kernel_size=2, stride=2)**:
- Takes 2×2×2 cube and keeps only the maximum value
- **Why**: Reduces spatial dimensions, computational cost, and overfitting
- **Output**: (batch, 32, 8, 64, 64)

**Visual Representation**:
```
Input:  [1, 16, 128, 128]
  ↓ Conv3D
[32, 16, 128, 128]
  ↓ MaxPool
[32, 8, 64, 64]
```

#### Convolutional Block 2
```python
nn.Conv3d(32, 64, kernel_size=3, padding=1)
nn.BatchNorm3d(64)
nn.ReLU(inplace=True)
nn.MaxPool3d(kernel_size=2, stride=2)
```
- Input: (batch, 32, 8, 64, 64)
- After Conv: (batch, 64, 8, 64, 64)
- After Pool: (batch, 64, 4, 32, 32)

#### Convolutional Block 3
```python
nn.Conv3d(64, 128, kernel_size=3, padding=1)
nn.BatchNorm3d(128)
nn.ReLU(inplace=True)
nn.MaxPool3d(kernel_size=2, stride=2)
```
- Input: (batch, 64, 4, 32, 32)
- After Conv: (batch, 128, 4, 32, 32)
- After Pool: (batch, 128, 2, 16, 16)

#### Convolutional Block 4
```python
nn.Conv3d(128, 256, kernel_size=3, padding=1)
nn.BatchNorm3d(256)
nn.ReLU(inplace=True)
nn.AdaptiveAvgPool3d((1, 1, 1))  # Global Average Pooling
```
- Input: (batch, 128, 2, 16, 16)
- After Conv: (batch, 256, 2, 16, 16)
- After **Adaptive** Pool: (batch, 256, 1, 1, 1)

**AdaptiveAvgPool3d((1, 1, 1))** - Global Average Pooling:
- Takes entire spatial volume and averages each channel to a single value
- **Why**: Reduces overfitting, makes model robust to spatial variations
- **Result**: 256 features per sample

#### Classifier (Fully Connected Layers)
```python
nn.Flatten()                    # (batch, 256, 1, 1, 1) → (batch, 256)
nn.Linear(256, 128)             # 256 → 128
nn.ReLU(inplace=True)
nn.Dropout(0.5)                 # Drop 50% of neurons
nn.Linear(128, 64)              # 128 → 64
nn.ReLU(inplace=True)
nn.Dropout(0.3)                 # Drop 30% of neurons
nn.Linear(64, 1)                # 64 → 1 (final prediction)
nn.Sigmoid()                    # Squash to [0, 1]
```

**Flatten**:
- Converts (batch, 256, 1, 1, 1) → (batch, 256)
- Prepares for fully connected layers

**Linear(256, 128)**:
- Fully connected layer: each output connected to all inputs
- Matrix multiplication: 256 inputs × 128 outputs = 32,768 parameters

**Dropout(0.5)**:
- Randomly sets 50% of neurons to zero during training
- **Why**: Prevents overfitting by forcing redundant learning
- **Only active during training**, turned off during evaluation

**Final Linear(64, 1)**:
- Produces single output value
- Raw value (logit) before sigmoid

**Sigmoid**:
- Activation: `σ(x) = 1 / (1 + e^(-x))`
- Squashes any value to range [0, 1]
- Interpreted as probability of ACL tear

### Model Summary

**Total Architecture**:
```
Input: (batch, 1, 16, 128, 128)
↓ Conv Block 1: (batch, 32, 8, 64, 64)
↓ Conv Block 2: (batch, 64, 4, 32, 32)
↓ Conv Block 3: (batch, 128, 2, 16, 16)
↓ Conv Block 4: (batch, 256, 1, 1, 1)
↓ Flatten: (batch, 256)
↓ FC Layers: (batch, 128) → (batch, 64) → (batch, 1)
Output: (batch, 1) - Probability in [0, 1]
```

**Parameter Count**:
- Convolutional layers: ~1.2M parameters
- Fully connected layers: ~50K parameters
- **Total**: ~1.25M trainable parameters

---

## 7. Training Process

### Data Splitting Strategy

#### Train-Validation-Test Split
```python
train_df, temp_df = train_test_split(valid_data, test_size=0.3, stratify=valid_data['binary_label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['binary_label'])
```

**Proportions**:
- **Training**: 70% (515 samples) - Used to train the model
- **Validation**: 15% (110 samples) - Used during training for hyperparameter tuning
- **Test**: 15% (111 samples) - Final evaluation, never seen during training

**Stratified Splitting**:
- `stratify=valid_data['binary_label']` ensures each split has the same class distribution
- **Example**: If original data is 75% No Tear / 25% Tear, all splits will maintain this ratio

**Why Three Splits?**:
- **Training**: Model learns patterns
- **Validation**: Monitor overfitting, tune hyperparameters
- **Test**: Unbiased final performance estimate

### DataLoader and Batching

```python
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
```

**Batch Size = 8**:
- Processes 8 MRI scans at once
- **Trade-off**:
  - Larger batch → more stable gradients, faster training, more GPU memory
  - Smaller batch → more updates, better generalization, less memory

**shuffle=True** (for training):
- Randomly reorders samples each epoch
- **Why**: Prevents model from learning the order of examples

**shuffle=False** (for validation/test):
- Consistent ordering for reproducible results

### Loss Function

#### Binary Cross-Entropy (BCE) Loss
```python
criterion = nn.BCELoss()
```

**Formula**:
```
BCE = -[y * log(ŷ) + (1-y) * log(1-ŷ)]

Where:
y  = true label (0 or 1)
ŷ  = predicted probability (0 to 1)
```

**Example**:
```
True label = 1 (ACL Tear)
Predicted = 0.9
BCE = -[1 * log(0.9) + 0 * log(0.1)] = 0.105 (low loss, good!)

True label = 1 (ACL Tear)
Predicted = 0.2
BCE = -[1 * log(0.2) + 0 * log(0.8)] = 1.609 (high loss, bad!)
```

**Why BCE for Binary Classification?**:
- Penalizes wrong predictions heavily
- Optimizes probability estimates
- Differentiable (allows backpropagation)

#### Alternative: Weighted BCE (for Imbalanced Data)
```python
pos_weight = torch.tensor([n_negative / n_positive])  # ~3.0
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```
- Gives 3× more weight to positive (tear) samples
- Compensates for class imbalance
- **Not used in final implementation** but mentioned in code

### Optimizer

#### Adam Optimizer
```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

**What Adam Does**:
- Updates model weights to minimize loss
- **Adaptive Learning Rates**: Different learning rate for each parameter
- **Momentum**: Uses past gradients to smooth updates

**Learning Rate (lr=0.001)**:
- Controls step size during optimization
- 0.001 = 10^-3 is a common default
- **Too high**: Training unstable, overshoots minimum
- **Too low**: Training very slow

**How Adam Works** (simplified):
```
1. Compute gradient: ∂Loss/∂weights
2. Update weights: weights -= lr * gradient (with momentum and adaptation)
3. Repeat for all batches
```

### Learning Rate Scheduler

```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)
```

**Purpose**: Automatically reduce learning rate when training plateaus

**Parameters**:
- **mode='min'**: Monitor validation loss (want to minimize it)
- **factor=0.5**: Multiply learning rate by 0.5 when plateau detected
- **patience=3**: Wait 3 epochs without improvement before reducing
- **verbose=True**: Print when learning rate changes

**Example Behavior**:
```
Epoch 1: val_loss=0.500, lr=0.001
Epoch 2: val_loss=0.450, lr=0.001
Epoch 3: val_loss=0.440, lr=0.001
Epoch 4: val_loss=0.441, lr=0.001 (no improvement)
Epoch 5: val_loss=0.442, lr=0.001 (no improvement)
Epoch 6: val_loss=0.443, lr=0.001 (no improvement)
Epoch 7: lr reduced to 0.0005 (3 epochs patience exceeded)
```

**Why This Helps**:
- Large learning rate for fast initial learning
- Smaller learning rate for fine-tuning later
- Automatic adaptation without manual intervention

### Training Loop

#### One Epoch Structure
```python
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()  # Enable dropout, batch norm in training mode

    for batch_x, batch_y in loader:
        # 1. Move data to GPU/CPU
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # 2. Zero gradients from previous iteration
        optimizer.zero_grad()

        # 3. Forward pass
        outputs = model(batch_x)

        # 4. Compute loss
        loss = criterion(outputs.squeeze(), batch_y)

        # 5. Backward pass (compute gradients)
        loss.backward()

        # 6. Update weights
        optimizer.step()
```

**Step-by-Step Explanation**:

**model.train()**:
- Sets model to training mode
- Enables dropout (randomly drops neurons)
- Enables batch normalization updates

**optimizer.zero_grad()**:
- Clears old gradients
- **Why**: PyTorch accumulates gradients by default

**Forward Pass**:
- Input flows through model
- Predictions are generated

**loss.backward()**:
- Computes gradients: ∂Loss/∂weights for all parameters
- Uses backpropagation algorithm
- Automatically handles complex chain rule

**optimizer.step()**:
- Updates all weights using computed gradients
- `weight = weight - learning_rate * gradient`

### Validation Loop

```python
def validate(model, loader, criterion, device):
    model.eval()  # Disable dropout, batch norm in eval mode

    with torch.no_grad():  # Disable gradient computation
        for batch_x, batch_y in loader:
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            predictions = (outputs > 0.5).float()
```

**Key Differences from Training**:

**model.eval()**:
- Disables dropout (uses all neurons)
- Batch norm uses running statistics instead of batch statistics

**with torch.no_grad()**:
- Disables gradient calculation
- **Why**: Saves memory and computation during evaluation

**No optimizer.step()**:
- Weights are not updated during validation

### Early Stopping

```python
PATIENCE = 7
patience_counter = 0

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

**Purpose**: Stop training when model stops improving

**How It Works**:
1. Monitor validation loss after each epoch
2. If validation loss improves, reset counter and save model
3. If validation loss doesn't improve, increment counter
4. If counter reaches PATIENCE (7 epochs), stop training

**Why Early Stopping?**:
- Prevents overfitting (model memorizing training data)
- Saves training time
- Automatically finds optimal number of epochs

**Example**:
```
Epoch 1: val_loss=0.50 ✓ (best, save model)
Epoch 2: val_loss=0.45 ✓ (best, save model)
Epoch 3: val_loss=0.47 ✗ (worse, counter=1)
Epoch 4: val_loss=0.46 ✗ (worse, counter=2)
...
Epoch 10: val_loss=0.48 ✗ (counter=7, STOP!)
Load best model from Epoch 2
```

### Complete Training Configuration

```python
NUM_EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 0.001
PATIENCE = 7
TARGET_SIZE = (16, 128, 128)
```

**These hyperparameters control**:
- Training duration (30 epochs max)
- Memory usage (batch size 8)
- Learning speed (lr 0.001)
- Overfitting prevention (patience 7)
- Input standardization (target size)

---

## 8. Evaluation Metrics

### Accuracy

```python
accuracy = correct_predictions / total_predictions
```

**Example**:
- 100 test samples
- 85 correctly predicted
- Accuracy = 85/100 = 0.85 = 85%

**Limitations**:
- Misleading with imbalanced data
- **Example**: If 90% are "No Tear", predicting always "No Tear" gives 90% accuracy but is useless

### Confusion Matrix

```
                Predicted
              No Tear  |  Tear
Actual -------------------------
No Tear   |    TN     |   FP
Tear      |    FN     |   TP
```

**Components**:
- **True Negative (TN)**: Correctly predicted No Tear
- **False Positive (FP)**: Predicted Tear, but actually No Tear (Type I error)
- **False Negative (FN)**: Predicted No Tear, but actually Tear (Type II error)
- **True Positive (TP)**: Correctly predicted Tear

**Example**:
```
              Predicted
           No Tear  |  Tear
Actual ----------------------
No Tear |    80     |   3
Tear    |    5      |   12
```
- TN=80, FP=3, FN=5, TP=12
- Total accuracy = (80+12)/100 = 92%

**Medical Significance**:
- **False Negative (FN)**: Most dangerous - miss a tear, patient doesn't get treatment
- **False Positive (FP)**: Less dangerous but causes unnecessary anxiety/procedures

### Precision, Recall, F1-Score

#### Precision
```python
Precision = TP / (TP + FP)
```
- **Question Answered**: "Of all predicted tears, how many were actually tears?"
- **Example**: TP=12, FP=3 → Precision = 12/15 = 0.80 = 80%
- **High Precision**: Few false alarms

#### Recall (Sensitivity)
```python
Recall = TP / (TP + FN)
```
- **Question Answered**: "Of all actual tears, how many did we detect?"
- **Example**: TP=12, FN=5 → Recall = 12/17 = 0.71 = 71%
- **High Recall**: Few missed cases

#### F1-Score
```python
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```
- **Harmonic mean** of precision and recall
- **Example**: Precision=0.80, Recall=0.71 → F1 = 2*(0.80*0.71)/(0.80+0.71) = 0.75
- **Balanced metric**: Good when you care equally about FP and FN

### Classification Report

```python
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred, target_names=['No Tear', 'ACL Tear']))
```

**Sample Output**:
```
              precision    recall  f1-score   support

     No Tear       0.94      0.96      0.95        83
    ACL Tear       0.80      0.71      0.75        28

    accuracy                           0.90       111
   macro avg       0.87      0.84      0.85       111
weighted avg       0.90      0.90      0.90       111
```

**Interpretation**:
- **support**: Number of samples in each class
- **macro avg**: Simple average across classes (treats classes equally)
- **weighted avg**: Weighted by class size (accounts for imbalance)

### ROC-AUC (Not Implemented, But Relevant)

**ROC Curve**: Plots True Positive Rate vs False Positive Rate at different thresholds

**AUC (Area Under Curve)**:
- Single number summarizing ROC curve
- **1.0**: Perfect classifier
- **0.5**: Random guessing
- **>0.8**: Generally considered good for medical AI

---

## 9. Technologies and Libraries

### PyTorch (torch)

**What**: Deep learning framework by Facebook/Meta

**Key Components Used**:

#### torch.nn (Neural Network Module)
```python
import torch.nn as nn
```
- `nn.Module`: Base class for all neural networks
- `nn.Conv3d`: 3D convolutional layer
- `nn.BatchNorm3d`: Batch normalization
- `nn.MaxPool3d`: Max pooling
- `nn.Linear`: Fully connected layer
- `nn.Dropout`: Dropout regularization
- `nn.ReLU`: ReLU activation
- `nn.Sigmoid`: Sigmoid activation
- `nn.BCELoss`: Binary cross-entropy loss

#### torch.optim (Optimization)
```python
import torch.optim as optim
```
- `optim.Adam`: Adam optimizer
- `optim.lr_scheduler.ReduceLROnPlateau`: Learning rate scheduler

#### torch.utils.data
```python
from torch.utils.data import Dataset, DataLoader
```
- `Dataset`: Abstract class for custom datasets
- `DataLoader`: Batching, shuffling, parallel loading

#### torch.device
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
- Automatically detects and uses GPU if available
- **CUDA**: NVIDIA's parallel computing platform
- **CPU fallback**: Works on any machine

### NumPy

```python
import numpy as np
```

**Uses**:
- Array operations on MRI data
- Mathematical operations
- Data type conversions
- Array slicing and indexing

**Key Functions Used**:
- `np.array()`: Create arrays
- `np.random.seed()`: Reproducibility
- Array operations: `.min()`, `.max()`, `.mean()`

### Pandas

```python
import pandas as pd
```

**Uses**:
- Load metadata CSV
- Data filtering and selection
- Statistical analysis
- Data splitting

**Key Functions Used**:
- `pd.read_csv()`: Load metadata
- `dataframe.iloc[]`: Row selection by index
- `dataframe['column']`: Column selection
- `.value_counts()`: Count class distribution

### scikit-image (skimage)

```python
from skimage.transform import resize
```

**Purpose**: Image processing

**Functions Used**:
- `resize()`: Resize 3D volumes with anti-aliasing
- Handles interpolation for smooth resizing

### scikit-learn (sklearn)

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
```

**Uses**:
- **train_test_split**: Split data into train/val/test sets
- **classification_report**: Compute precision, recall, F1
- **confusion_matrix**: Generate confusion matrix
- **accuracy_score**: Calculate accuracy

### Matplotlib

```python
import matplotlib.pyplot as plt
```

**Uses**:
- Visualize MRI slices
- Plot training curves
- Display confusion matrix
- Show predictions

**Key Functions**:
- `plt.imshow()`: Display images
- `plt.plot()`: Line plots
- `plt.subplot()`: Multiple plots
- `plt.savefig()`: Save figures

### Python Standard Library

#### pickle
```python
import pickle
```
- Serialize/deserialize Python objects
- Load .pck MRI files

#### os
```python
import os
```
- File path operations
- `os.path.join()`: Platform-independent path joining
- `os.path.exists()`: Check file existence

#### collections.Counter
```python
from collections import Counter
```
- Count class distributions

---

## 10. File Structure

### Project Root
```
ACL tears/
├── .git/                          # Git repository
├── .gitignore                     # Git ignore rules
├── .gitattributes                 # Git attributes
├── LICENSE                        # Project license
├── README.md                      # Project overview
│
├── mri_env/                       # Virtual environment
│   └── ...                        # Python packages
│
├── DATASET/                       # Data directory (not in repo)
│   └── MRI/
│       ├── metadata.csv          # Labels and ROI info
│       ├── vol01/                # Volume folders
│       ├── vol02/
│       ├── ...
│       └── vol08/
│
├── notebook/                      # Development notebooks
│   ├── 01_test.ipynb             # PyTorch testing
│   ├── 03_dataset_pipeline.ipynb # Dataset experiments
│   └── test.py                   # Simple PyTorch test
│
├── ACL_Tear_Detection_Complete.ipynb  # Main training notebook (in checkpoints)
├── 04_test_dataset.ipynb              # Dataset testing
│
├── acl_detector_model.pth         # Trained model weights (~14MB)
├── training_history.png           # Training curves
├── confusion_matrix.png           # Performance visualization
└── predictions_visualization.png  # Sample predictions
```

### Key Files Explained

#### ACL_Tear_Detection_Complete.ipynb
- **Main training notebook** (736 lines)
- Contains complete pipeline from data loading to evaluation
- Well-documented with markdown explanations
- Designed for educational purposes

#### acl_detector_model.pth
- PyTorch model checkpoint (~14MB)
- Contains:
  - `model_state_dict`: Trained weights
  - `optimizer_state_dict`: Optimizer state
  - `history`: Training/validation metrics
  - `best_val_loss`: Best validation loss achieved

**Loading**:
```python
checkpoint = torch.load('acl_detector_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

#### metadata.csv
- CSV file with 917 rows (one per MRI scan)
- 11 columns (examId, aclDiagnosis, ROI coordinates, etc.)
- Critical for mapping files to labels

#### Visualization Files
- **training_history.png**: Loss and accuracy curves over epochs
- **confusion_matrix.png**: 2×2 matrix showing predictions vs actual
- **predictions_visualization.png**: 6 sample predictions with ROI images

---

## 11. Key Concepts Explained

### 1. Binary Classification

**Definition**: Categorizing inputs into one of two classes

**In This Project**:
- **Class 0**: No ACL Tear (negative class)
- **Class 1**: ACL Tear (positive class)

**Why Binary Instead of Multi-class?**:
- Simplifies the problem (3 classes → 2 classes)
- More balanced (227 tears vs 690 no tears is better than 690/150/77 split)
- Clinically relevant (tear vs no tear is the primary diagnostic question)

**Converting Multi-class to Binary**:
```python
metadata['binary_label'] = (metadata['aclDiagnosis'] > 0).astype(int)
# 0 → 0 (no tear)
# 1 → 1 (partial tear → tear)
# 2 → 1 (complete tear → tear)
```

### 2. Class Imbalance

**Problem**: Unequal number of samples per class

**In This Project**:
- No Tear: 690 samples (75%)
- Tear: 227 samples (25%)
- **Imbalance Ratio**: 3:1

**Why It Matters**:
- Model may bias toward majority class
- High accuracy by always predicting "No Tear"
- Poor performance on minority class (the important one!)

**Solutions**:
1. **Weighted Loss**: Give more weight to minority class
2. **Data Augmentation**: Generate more minority samples
3. **Stratified Splitting**: Maintain class ratio in all splits
4. **Evaluation Metrics**: Use F1, precision, recall instead of just accuracy

### 3. Overfitting vs Underfitting

#### Overfitting
- **Problem**: Model memorizes training data, fails on new data
- **Signs**:
  - Training accuracy high (>95%)
  - Validation accuracy low (<70%)
  - Large gap between train and val loss
- **Solutions**:
  - Dropout (randomly disable neurons)
  - Early stopping
  - More training data
  - Data augmentation

#### Underfitting
- **Problem**: Model too simple, can't learn patterns
- **Signs**:
  - Both training and validation accuracy low
  - Model performs poorly everywhere
- **Solutions**:
  - Larger model (more layers/parameters)
  - Train longer
  - Better features

**Ideal**: Balanced model that generalizes well

### 4. Convolution Operation

**1D Example**:
```
Input:  [1, 2, 3, 4, 5]
Kernel: [0.5, 1, 0.5]

Output[0] = 0.5*1 + 1*2 + 0.5*3 = 3.5
Output[1] = 0.5*2 + 1*3 + 0.5*4 = 5.5
Output[2] = 0.5*3 + 1*4 + 0.5*5 = 7.5
```

**3D Convolution**:
- Kernel is a 3D cube (e.g., 3×3×3)
- Slides over volume in all three dimensions
- Produces 3D output feature maps

**Purpose**:
- Detect patterns (edges, textures, shapes)
- Learn hierarchical features
- Translation invariant (detects pattern anywhere in image)

### 5. Pooling

**Max Pooling Example** (2×2):
```
Input:
[1 2]
[3 4]

Output: 4 (maximum value)
```

**Purpose**:
- **Dimension Reduction**: Reduce spatial size
- **Computational Efficiency**: Fewer parameters in next layer
- **Translation Invariance**: Small shifts don't affect output
- **Feature Selection**: Keep most prominent features

**Types**:
- **Max Pooling**: Take maximum (used in early layers)
- **Average Pooling**: Take average
- **Global Average Pooling**: Pool entire feature map to 1 value (used in last conv block)

### 6. Batch Normalization

**Problem**: Internal covariate shift (layer inputs change during training)

**Solution**: Normalize layer inputs to have mean=0, std=1

**Formula**:
```
x_normalized = (x - μ_batch) / √(σ²_batch + ε)
output = γ * x_normalized + β
```

Where:
- μ_batch, σ²_batch: Batch mean and variance
- γ, β: Learnable parameters
- ε: Small constant (1e-5) for numerical stability

**Benefits**:
- Faster training
- Higher learning rates possible
- Less sensitive to weight initialization
- Regularization effect (slight noise from batch statistics)

### 7. Dropout

**Concept**: Randomly set neurons to zero during training

**Implementation**:
```python
nn.Dropout(0.5)  # 50% dropout rate
```

**Example**:
```
Layer output: [0.5, 0.8, 0.3, 0.9]
After dropout: [0.0, 0.8, 0.0, 0.9]  # 50% randomly zeroed
```

**Why It Works**:
- Prevents co-adaptation of neurons
- Forces redundant representations
- Ensemble effect (different subnetworks each iteration)

**During Evaluation**: Dropout is disabled, all neurons used

### 8. Backpropagation

**Purpose**: Compute gradients for all parameters

**Chain Rule**:
```
∂Loss/∂w₁ = (∂Loss/∂output) × (∂output/∂w₁)
```

**Process**:
1. Forward pass: Compute predictions and loss
2. Backward pass: Compute gradients layer-by-layer (from output to input)
3. Update weights using gradients

**PyTorch Automation**:
```python
loss.backward()  # Automatically computes all gradients
optimizer.step() # Updates all weights
```

### 9. Gradient Descent

**Concept**: Iteratively update weights to minimize loss

**Update Rule**:
```
weight_new = weight_old - learning_rate × gradient
```

**Variants**:
- **Batch GD**: Use all data (slow)
- **Stochastic GD**: Use one sample (noisy)
- **Mini-batch GD**: Use small batch (best of both)

**Adam Optimizer**: Advanced variant with:
- Adaptive learning rates per parameter
- Momentum (smooths updates)
- Bias correction

### 10. Transfer Learning (Not Used Here)

**Concept**: Use pre-trained model on similar task

**Why Relevant**:
- Medical imaging datasets are often small
- Pre-trained models (e.g., on ImageNet) provide good feature extractors
- Fine-tune final layers on medical data

**Potential Improvement**:
```python
# Use pre-trained 3D ResNet
from torchvision.models.video import r3d_18
model = r3d_18(pretrained=True)
# Replace final layer for binary classification
model.fc = nn.Linear(512, 1)
```

### 11. Tensor Operations

**Tensor**: Multi-dimensional array (generalization of matrices)

**Dimensions**:
- **Scalar**: 0D tensor (single number)
- **Vector**: 1D tensor [1, 2, 3]
- **Matrix**: 2D tensor [[1,2], [3,4]]
- **3D Tensor**: [[[1,2], [3,4]], [[5,6], [7,8]]]
- **4D Tensor**: Common in images (batch, channels, height, width)
- **5D Tensor**: Used here (batch, channels, depth, height, width)

**Key Operations**:
```python
x.unsqueeze(0)   # Add dimension
x.squeeze()      # Remove dimension
x.permute(dims)  # Reorder dimensions
x.view(shape)    # Reshape (must be contiguous)
x.to(device)     # Move to GPU/CPU
```

### 12. Activation Functions

#### ReLU (Rectified Linear Unit)
```
f(x) = max(0, x)
```
- **Pros**: Fast, no vanishing gradient, sparse activation
- **Cons**: Dead neurons (if weights push all inputs < 0)

#### Sigmoid
```
f(x) = 1 / (1 + e^(-x))
```
- **Output**: (0, 1) - interpreted as probability
- **Pros**: Smooth, differentiable
- **Cons**: Vanishing gradient for large |x|

**Why ReLU in Hidden Layers, Sigmoid in Output?**:
- ReLU: Better gradient flow, faster training
- Sigmoid: Probability output for binary classification

### 13. GPU Acceleration

**Why Use GPU?**:
- **Parallel Processing**: GPUs have thousands of cores
- **Matrix Operations**: Deep learning is mostly matrix multiplication
- **Speed**: 10-100× faster than CPU

**PyTorch GPU Usage**:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)      # Move model to GPU
data = data.to(device)        # Move data to GPU
```

**Memory Consideration**:
- GPUs have limited memory (8-24GB typical)
- Batch size limited by GPU memory
- Larger models require more memory

### 14. Model Saving and Loading

**Saving**:
```python
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'history': history
}, 'model.pth')
```

**Loading**:
```python
checkpoint = torch.load('model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set to evaluation mode
```

**state_dict**: Dictionary mapping parameter names to tensors

### 15. Random Seeds for Reproducibility

```python
np.random.seed(42)
torch.manual_seed(42)
```

**Purpose**: Make results reproducible

**Why**:
- Random initialization of weights
- Random data shuffling
- Random dropout masks
- Setting seed ensures same "randomness" every run

**Note**: Perfect reproducibility on GPU is harder (CUDA operations may not be deterministic)

---

## Conclusion

This ACL tear detection project demonstrates a complete deep learning pipeline for medical image analysis:

1. **Data Handling**: Loading, preprocessing, and organizing 3D MRI data
2. **Model Architecture**: Custom 3D CNN with 4 convolutional blocks
3. **Training Strategy**: Proper splitting, early stopping, learning rate scheduling
4. **Evaluation**: Comprehensive metrics including confusion matrix and classification report
5. **Deployment Ready**: Saved model with prediction functions

**Key Takeaways**:
- Medical imaging requires specialized preprocessing (ROI extraction, normalization)
- 3D CNNs are essential for volumetric medical data
- Class imbalance must be addressed in medical datasets
- Proper evaluation metrics (beyond accuracy) are critical
- Early stopping prevents overfitting

**Potential Applications**:
- Assist radiologists in preliminary screening
- Reduce diagnosis time
- Second opinion system
- Educational tool for medical students

**Important Disclaimer**: This is a research/educational project and should NOT be used for actual medical diagnosis without proper clinical validation, regulatory approval, and oversight by medical professionals.

---

**End of Documentation**
