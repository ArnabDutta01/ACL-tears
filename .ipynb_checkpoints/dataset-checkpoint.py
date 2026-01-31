import os
import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
from skimage.transform import resize
#helper
def load_pck(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def extract_roi(mri, row):
    z, y, x = row["roiZ"], row["roiY"], row["roiX"]
    d, h, w = row["roiDepth"], row["roiHeight"], row["roiWidth"]
    return mri[z:z+d, y:y+h, x:x+w]

def normalize(volume):
    return (volume - volume.min()) / (volume.max() - volume.min() + 1e-6)

def resize_roi(roi, target_shape=(8, 128, 128)):
    return resize(roi, target_shape, mode=
                  class KneeDataset(Dataset):
    def __init__(self, df, data_dir):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        pck_path = os.path.join(self.data_dir, row["volumeFilename"])
        mri = load_pck(pck_path)

        roi = extract_roi(mri, row)
        roi = normalize(roi)
        roi = resize_roi(roi)

        roi = torch.tensor(roi, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(row["aclDiagnosis"], dtype=torch.float32)

        return roi, label
"constant", anti_aliasing=True)
#kneeclass