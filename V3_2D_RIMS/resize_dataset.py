"""
Resize processed sagittal MRI volumes for Google Colab.
Reduces 560x560+ images to 224x224 and saves as compressed uint8.
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import csv
from skimage.transform import resize
import shutil

# Configuration
INPUT_DIR = Path(r"c:\ML projects\ACL tears\DATASET\processed_sagittal")
OUTPUT_DIR = Path(r"c:\ML projects\ACL tears\DATASET\processed_sagittal_resized")
TARGET_SIZE = (224, 224)  # Standard size for many CNN architectures

def resize_volume(volume, target_size):
    """
    Resize a 3D volume (slices, H, W) to (slices, target_H, target_W).
    """
    num_slices = volume.shape[0]
    resized_slices = []
    
    for i in range(num_slices):
        # Resize each slice
        resized = resize(volume[i], target_size, mode='reflect', 
                        anti_aliasing=True, preserve_range=True)
        resized_slices.append(resized)
    
    return np.stack(resized_slices, axis=0)


def normalize_to_uint8(volume):
    """
    Normalize volume to 0-255 uint8 range.
    """
    v_min, v_max = volume.min(), volume.max()
    if v_max - v_min > 0:
        volume = (volume - v_min) / (v_max - v_min) * 255
    return volume.astype(np.uint8)


def process_dataset():
    """
    Resize all volumes and save as compressed npz.
    """
    print("=" * 60)
    print("Resizing ACL Dataset for Google Colab")
    print(f"Target size: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")
    print("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Read metadata
    metadata_file = INPUT_DIR / "metadata.csv"
    with open(metadata_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        metadata = list(reader)
    
    print(f"\nFound {len(metadata)} volumes to process")
    
    # Process each volume
    new_metadata = []
    
    for row in tqdm(metadata, desc="Resizing"):
        input_path = INPUT_DIR / row['filename']
        
        if not input_path.exists():
            continue
        
        # Load volume
        volume = np.load(input_path)
        
        # Resize
        resized = resize_volume(volume, TARGET_SIZE)
        
        # Convert to uint8
        resized_uint8 = normalize_to_uint8(resized)
        
        # Save as compressed npz
        output_filename = row['filename'].replace('.npy', '.npz')
        output_path = OUTPUT_DIR / output_filename
        np.savez_compressed(output_path, data=resized_uint8)
        
        # Update metadata
        new_row = row.copy()
        new_row['filename'] = output_filename
        new_row['height'] = TARGET_SIZE[0]
        new_row['width'] = TARGET_SIZE[1]
        new_metadata.append(new_row)
    
    # Save new metadata
    new_metadata_file = OUTPUT_DIR / "metadata.csv"
    with open(new_metadata_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=new_metadata[0].keys())
        writer.writeheader()
        writer.writerows(new_metadata)
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*.npz"))
    
    print(f"\nProcessing complete!")
    print(f"  Volumes processed: {len(new_metadata)}")
    print(f"  Total output size: {total_size / (1024**2):.1f} MB")
    print(f"  Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    process_dataset()
