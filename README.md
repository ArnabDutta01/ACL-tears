# ACL-tears
My first projects to detect ACL tears on MRI scan.
# ACL Tear Detection from MRI Scans

A deep learning project that uses 3D Convolutional Neural Networks (CNN) to automatically detect Anterior Cruciate Ligament (ACL) tears from knee MRI scans.

## 🎯 Project Overview

This project implements a medical imaging AI system that can classify knee MRI scans into three categories:
- **0**: No ACL tear (healthy)
- **1**: Partial ACL tear
- **2**: Complete ACL tear

The model achieves this by processing 3D MRI volumes and learning to identify patterns associated with ACL injuries.

## 🏗️ Model Architecture

- **Type**: 3D Convolutional Neural Network (CNN)
- **Framework**: PyTorch 2.9.1
- **Architecture**:
  - 4 convolutional blocks with increasing filters (32 → 64 → 128 → 256)
  - Batch normalization and ReLU activations
  - Global average pooling
  - Fully connected layers with dropout (0.5 and 0.3)
  - Binary classification output (Tear vs No Tear)

## 📊 Dataset

- **Total samples**: 917 MRI scans
- **Available files**: 736 scans
- **Format**: 3D MRI volumes stored as `.pck` files
- **Input size**: Variable (resized to 16×128×128)
- **Class distribution**:
  - No Tear: 690 samples (75%)
  - Partial/Complete Tear: 227 samples (25%)

### Data Split
- Training: 70% (515 samples)
- Validation: 15% (110 samples)
- Test: 15% (111 samples)

**Note**: The dataset is NOT included in this repository due to size (~5GB) and privacy concerns. You'll need to provide your own MRI dataset.

## 📁 Project Structure

```
ACL-tears/
├── ACL_Tear_Detection_Complete.ipynb  # Main training notebook
├── dataset.py                          # Dataset class implementation
├── requirements.txt                    # Python dependencies
├── notebook/                           # Development notebooks
│   ├── 03_dataset_pipeline.ipynb
│   └── numpy_prac1.ipynb
├── 04_test_dataset.ipynb              # Testing pipeline
├── test_new_data.ipynb                # Inference on new data
├── DATASET/                            # [NOT INCLUDED] MRI data
├── acl_detector_model.pth             # [NOT INCLUDED] Trained model
└── *.png                               # [NOT INCLUDED] Generated visualizations
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/acl-tear-detection.git
cd acl-tear-detection
```

2. Create a virtual environment:
```bash
python -m venv mri_env
source mri_env/bin/activate  # On Windows: mri_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dataset Setup

1. Place your MRI dataset in the `DATASET/MRI/` folder
2. Organize files in `vol01/` through `vol08/` subfolders
3. Ensure `metadata.csv` contains the following columns:
   - `examId`, `seriesNo`, `aclDiagnosis`, `kneeLR`
   - `roiX`, `roiY`, `roiZ`, `roiHeight`, `roiWidth`, `roiDepth`
   - `volumeFilename`

### Training

Open and run the main notebook:
```bash
jupyter notebook ACL_Tear_Detection_Complete.ipynb
```

Or train programmatically by following the notebook cells step by step.

## 📈 Results

The model generates the following visualizations:
- **Training History**: Loss and accuracy curves over epochs
- **Confusion Matrix**: Classification performance on test set
- **Predictions Visualization**: Sample predictions with confidence scores

## 🔧 Key Components

### Dataset Class (`dataset.py`)
- Loads 3D MRI volumes from `.pck` files
- Extracts Region of Interest (ROI) using metadata coordinates
- Normalizes and resizes volumes to consistent dimensions
- Returns PyTorch tensors ready for training

### Model Features
- Handles 3D medical imaging data
- Early stopping to prevent overfitting
- Learning rate scheduling (ReduceLROnPlateau)
- Class imbalance handling with weighted loss

## 📝 Usage Example

```python
import torch
from dataset import KneeDataset
import pandas as pd

# Load metadata
metadata = pd.read_csv('DATASET/MRI/metadata.csv')

# Create dataset
dataset = KneeDataset(metadata, 'DATASET/MRI')

# Load trained model
model = torch.load('acl_detector_model.pth')
model.eval()

# Make prediction
sample, label = dataset[0]
with torch.no_grad():
    prediction = model(sample.unsqueeze(0))
    print(f"Prediction: {prediction.item():.2%} confidence of ACL tear")
```

## 🛠️ Technologies Used

- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **scikit-image**: Image preprocessing
- **scikit-learn**: Model evaluation metrics
- **Matplotlib**: Visualization
- **Jupyter**: Interactive development

## ⚠️ Important Notes

1. **Medical Disclaimer**: This is a research/educational project. Do NOT use for actual medical diagnosis without proper validation and regulatory approval.

2. **Data Privacy**: Never commit actual MRI data to version control. Ensure compliance with HIPAA and other medical data regulations.

3. **Model Files**: The trained model (`*.pth`) is excluded from git due to size. Consider using Git LFS or cloud storage for sharing models.

## 🔮 Future Improvements

- [ ] Data augmentation (rotation, flipping)
- [ ] Transfer learning with pre-trained 3D models
- [ ] Multi-class classification (partial vs complete tears)
- [ ] Attention mechanisms for interpretability
- [ ] Cross-validation for robust evaluation
- [ ] Deployment as a web service

## 📄 License

[Choose appropriate license - e.g., MIT, Apache 2.0]

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

[Your Name/Contact Information]

## 🙏 Acknowledgments

- MRI dataset source [if applicable]
- Research papers that influenced this work
- Open-source community

---

**Note**: This project is for educational and research purposes only.
