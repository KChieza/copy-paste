import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm

def filter_masks(input_folder, output_folder, min_size_kb):
    """
    Filters masks based on file size.
    
    Args:
        input_folder (str): Path to the folder containing mask images.
        output_folder (str): Path to save the filtered mask images.
        min_size_kb (int): Minimum file size in KB for a valid mask.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    mask_files = [f for f in os.listdir(input_folder) if f.endswith('.png') or f.endswith('.jpg')]
    
    for mask_file in tqdm(mask_files, desc="Filtering masks by size"):
        mask_path = os.path.join(input_folder, mask_file)
        file_size_kb = os.path.getsize(mask_path) / 1024  # Convert bytes to KB
        
        if file_size_kb >= min_size_kb:
            output_path = os.path.join(output_folder, mask_file)
            shutil.copy(mask_path, output_path)
    
    print(f"Filtering complete. Valid masks saved in {output_folder}")

# Example usage
filter_masks("masks_iou10", "masks_iou10_bysize", min_size_kb)

def filter_masks_by_components(input_folder, output_folder, max_components):
    """
    Filters masks based on the number of connected components.
    
    Args:
        input_folder (str): Path to the folder containing mask images.
        output_folder (str): Path to save the filtered and rejected mask images.
        max_components (int): Maximum allowed connected components.
    """
    passed_folder = os.path.join(output_folder, "passed")
    failed_folder = os.path.join(output_folder, "failed")
    
    os.makedirs(passed_folder, exist_ok=True)
    os.makedirs(failed_folder, exist_ok=True)
    
    mask_files = [f for f in os.listdir(input_folder) if f.endswith('.png') or f.endswith('.jpg')]
    
    for mask_file in tqdm(mask_files, desc="Filtering masks by components"):
        mask_path = os.path.join(input_folder, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        num_components = num_labels - 1  # Exclude background
        
        if num_components <= max_components:
            shutil.copy(mask_path, os.path.join(passed_folder, mask_file))
        else:
            shutil.copy(mask_path, os.path.join(failed_folder, mask_file))
    
    print(f"Filtering complete. Passed masks saved in {passed_folder}, failed masks saved in {failed_folder}")

# Example usage
filter_masks_by_components("masks_iou10_bysize", "masks_iou10_bysize_by5componets", max_components=5)
