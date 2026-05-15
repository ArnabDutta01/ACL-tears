"""
DICOM to NumPy Preprocessing Script for ACL Tear Detection
Extracts sagittal MRI views from Priyank Saxena's ACL dataset and converts to NumPy arrays.

NOTE: This dataset uses multi-frame DICOM files where each file contains all slices of a series.
"""

import os
import numpy as np
import pydicom
from pathlib import Path
from collections import defaultdict
import csv
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATASET_ROOT = Path(r"c:\ML projects\ACL tears\DATASET\Priyank Saxena's files - ACL")
OUTPUT_DIR = Path(r"c:\ML projects\ACL tears\DATASET\processed_sagittal")
METADATA_FILE = OUTPUT_DIR / "metadata.csv"

# Label mapping
LABEL_MAP = {
    "NORMAL ACL": 0,
    "PARTIAL TEAR": 1,
    "COMPLETE TEAR": 2
}


def is_sagittal_series(series_desc):
    """Check if series description indicates sagittal plane."""
    if not series_desc:
        return False
    return 'sag' in series_desc.lower()


def find_sagittal_files(patient_path):
    """
    Find multi-frame DICOM files with sagittal MRI data.
    
    Returns list of (file_path, series_desc, num_frames, shape) for sagittal series.
    """
    sagittal_files = []
    
    for dcm_file in patient_path.glob("*.dcm"):
        try:
            # First quick check with metadata only
            ds = pydicom.dcmread(dcm_file, stop_before_pixels=True)
            
            series_desc = getattr(ds, 'SeriesDescription', '')
            num_frames = int(getattr(ds, 'NumberOfFrames', 1))
            
            # Only include multi-frame sagittal series
            if num_frames > 1 and is_sagittal_series(series_desc):
                # Get dimensions from metadata
                rows = int(getattr(ds, 'Rows', 0))
                cols = int(getattr(ds, 'Columns', 0))
                
                # Verify file has actual pixel data by checking file size
                # Multi-frame files with pixel data are typically > 1MB
                if dcm_file.stat().st_size > 100000:  # > 100KB
                    sagittal_files.append((dcm_file, series_desc, num_frames, (num_frames, rows, cols)))
        except Exception as e:
            continue
    
    return sagittal_files


def select_best_sagittal_file(sagittal_files):
    """
    Select the best sagittal series for ACL analysis.
    Preference: PD-weighted > T1-weighted > FFE, with more slices preferred.
    """
    if not sagittal_files:
        return None
    
    def score_series(item):
        file_path, series_desc, num_frames, shape = item
        desc_lower = series_desc.lower()
        
        # Scoring: higher is better
        score = num_frames  # Base score is number of slices
        
        # Prefer PD (Proton Density) weighted - good for soft tissue
        if 'pd' in desc_lower:
            score += 100
        # T1 weighted is also useful
        elif 't1' in desc_lower:
            score += 50
        # FFE is okay
        elif 'ffe' in desc_lower:
            score += 25
        
        # Prefer non-enhanced/non-SENSE versions (original quality)
        if not desc_lower.startswith('e') and 'sense' not in desc_lower:
            score += 10
        
        return score
    
    return max(sagittal_files, key=score_series)


def load_multiframe_volume(file_path):
    """
    Load multi-frame DICOM file as 3D numpy array.
    """
    ds = pydicom.dcmread(file_path)
    pixel_array = ds.pixel_array.astype(np.float32)
    
    # Apply rescale if available
    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
        pixel_array = pixel_array * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    
    return pixel_array


def normalize_volume(volume):
    """Normalize volume to 0-1 range."""
    if volume is None:
        return None
    
    v_min, v_max = volume.min(), volume.max()
    if v_max - v_min > 0:
        volume = (volume - v_min) / (v_max - v_min)
    return volume


def get_label_from_path(patient_path):
    """
    Determine label from folder structure.
    Primary label comes from parent folder (COMPLETE TEAR, PARTIAL TEAR, NORMAL ACL).
    """
    path_str = str(patient_path)
    path_parts = path_str.replace('\\', '/').upper().split('/')
    
    # Check parent directories for main category
    if 'NORMAL ACL' in path_parts:
        return 0, "NORMAL"
    elif 'COMPLETE TEAR' in path_parts:
        return 2, "COMPLETE"
    elif 'PARTIAL TEAR' in path_parts:
        return 1, "PARTIAL"
    
    # Fallback: check folder name itself with word boundaries
    folder_name = patient_path.name.upper()
    
    if 'NORMAL' in folder_name:
        return 0, "NORMAL"
    
    # Check for ACL grade patterns (with space to avoid matching numbers like 021)
    import re
    if re.search(r'\bACL\s*3\b|\b3\s*ACL\b', folder_name):
        return 2, "COMPLETE"
    elif re.search(r'\bACL\s*[12]\b|\b[12]\s*ACL\b|\bPARTIAL\b', folder_name):
        return 1, "PARTIAL"
    
    return -1, "UNKNOWN"


def get_all_patient_folders():
    """
    Get all patient folders with their labels.
    Returns list of (patient_path, label, label_name)
    """
    patients = []
    
    # Normal ACL patients
    normal_dir = DATASET_ROOT / "NORMAL ACL"
    if normal_dir.exists():
        for patient_folder in normal_dir.iterdir():
            if patient_folder.is_dir():
                patients.append((patient_folder, 0, "NORMAL"))
    
    # ACL Tear - Complete
    complete_dir = DATASET_ROOT / "ACL TEAR" / "COMPLETE TEAR"
    if complete_dir.exists():
        for patient_folder in complete_dir.iterdir():
            if patient_folder.is_dir():
                label, label_name = get_label_from_path(patient_folder)
                patients.append((patient_folder, label, label_name))
    
    # ACL Tear - Partial
    partial_dir = DATASET_ROOT / "ACL TEAR" / "PARTIAL TEAR"
    if partial_dir.exists():
        for patient_folder in partial_dir.iterdir():
            if patient_folder.is_dir():
                label, label_name = get_label_from_path(patient_folder)
                patients.append((patient_folder, label, label_name))
    
    return patients


def process_dataset(dry_run=False, sample_n=None):
    """
    Process the entire dataset.
    
    Args:
        dry_run: If True, just analyze without saving
        sample_n: If set, only process first N patients
    """
    print("=" * 60)
    print("ACL DICOM to NumPy Preprocessing (Multi-frame DICOM)")
    print("=" * 60)
    
    # Get all patients
    patients = get_all_patient_folders()
    print(f"\nFound {len(patients)} patient folders")
    
    # Count labels
    label_counts = defaultdict(int)
    for _, label, label_name in patients:
        label_counts[label_name] += 1
    print(f"Label distribution: {dict(label_counts)}")
    
    if sample_n:
        patients = patients[:sample_n]
        print(f"\nProcessing only {sample_n} samples")
    
    if not dry_run:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process each patient
    metadata_rows = []
    success_count = 0
    fail_count = 0
    
    for patient_path, label, label_name in tqdm(patients, desc="Processing"):
        patient_id = patient_path.name
        
        # Find sagittal multi-frame files
        sagittal_files = find_sagittal_files(patient_path)
        
        if not sagittal_files:
            if dry_run:
                print(f"  No sagittal series found: {patient_id}")
            fail_count += 1
            continue
        
        # Select best sagittal file
        best_file = select_best_sagittal_file(sagittal_files)
        file_path, series_desc, num_frames, shape = best_file
        
        if dry_run:
            print(f"  {patient_id}: {series_desc} ({num_frames} slices, {shape[1]}x{shape[2]}) -> {label_name}")
            success_count += 1
            continue
        
        # Load volume
        try:
            volume = load_multiframe_volume(file_path)
        except Exception as e:
            fail_count += 1
            continue
        
        if volume is None:
            fail_count += 1
            continue
        
        # Normalize
        volume = normalize_volume(volume)
        
        # Save as compressed numpy
        output_filename = f"{patient_id.replace(' ', '_')}.npy"
        output_path = OUTPUT_DIR / output_filename
        np.save(output_path, volume.astype(np.float16))  # float16 to save space
        
        # Record metadata
        metadata_rows.append({
            'patient_id': patient_id,
            'filename': output_filename,
            'label': label,
            'label_name': label_name,
            'num_slices': volume.shape[0],
            'height': volume.shape[1],
            'width': volume.shape[2],
            'series': series_desc
        })
        
        success_count += 1
    
    print(f"\n{'Analysis' if dry_run else 'Processing'} complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed/Skipped: {fail_count}")
    
    if not dry_run and metadata_rows:
        # Save metadata CSV
        with open(METADATA_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=metadata_rows[0].keys())
            writer.writeheader()
            writer.writerows(metadata_rows)
        print(f"\nMetadata saved to: {METADATA_FILE}")
        
        # Print final stats
        total_size = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*.npy"))
        print(f"Total output size: {total_size / (1024**2):.1f} MB")


def analyze_single_patient(patient_path):
    """
    Analyze DICOM metadata for a single patient (for debugging).
    """
    print(f"\nAnalyzing: {patient_path}")
    print("-" * 60)
    
    patient_path = Path(patient_path)
    
    for dcm_file in sorted(patient_path.glob("*.dcm")):
        try:
            ds = pydicom.dcmread(dcm_file)
            
            # Skip files without pixel data
            if not hasattr(ds, 'PixelData'):
                print(f"{dcm_file.name}: Metadata only (no pixel data)")
                continue
            
            series_desc = getattr(ds, 'SeriesDescription', 'N/A')
            num_frames = int(getattr(ds, 'NumberOfFrames', 1))
            is_sag = is_sagittal_series(series_desc)
            
            print(f"\n{dcm_file.name}:")
            print(f"  Series: {series_desc}")
            print(f"  Frames: {num_frames}")
            print(f"  Shape: {ds.pixel_array.shape}")
            print(f"  Sagittal: {is_sag}")
            
        except Exception as e:
            print(f"  Error reading {dcm_file.name}: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess ACL DICOM dataset")
    parser.add_argument('--dry-run', action='store_true', 
                        help='Analyze without saving files')
    parser.add_argument('--sample', type=int, default=None,
                        help='Process only first N patients')
    parser.add_argument('--analyze', type=str, default=None,
                        help='Analyze a specific patient folder')
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_single_patient(args.analyze)
    else:
        process_dataset(dry_run=args.dry_run, sample_n=args.sample)
