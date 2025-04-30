import os
import cv2
import numpy as np
import shutil
import argparse
from tqdm import tqdm

def filter_masks(input_folder, output_folder, min_size_kb=None, max_components=None):
    """
    Filters masks based on file size and/or number of connected components.
    Maintains consistent passed/failed folder structure for all filter types.
    
    Args:
        input_folder (str): Path to the folder containing mask images.
        output_folder (str): Path to save the filtered mask images.
        min_size_kb (int, optional): Minimum file size in KB for a valid mask.
        max_components (int, optional): Maximum allowed connected components.
    """
    if min_size_kb is None and max_components is None:
        raise ValueError("At least one filtering criterion (min_size_kb or max_components) must be provided")
    
    # Create consistent output folder structure
    passed_folder = os.path.join(output_folder, "passed")
    failed_folder = os.path.join(output_folder, "failed")
    os.makedirs(passed_folder, exist_ok=True)
    os.makedirs(failed_folder, exist_ok=True)
    
    mask_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for mask_file in tqdm(mask_files, desc="Filtering masks"):
        mask_path = os.path.join(input_folder, mask_file)
        passed = True
        
        # Apply size filter if specified
        if min_size_kb is not None:
            file_size_kb = os.path.getsize(mask_path) / 1024  # Convert bytes to KB
            if file_size_kb < min_size_kb:
                passed = False
        
        # Apply component filter if specified
        if max_components is not None and passed:  # Only check if still passing
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                passed = False
            else:
                # Find connected components
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
                num_components = num_labels - 1  # Exclude background
                
                if num_components > max_components:
                    passed = False
        
        # Copy to appropriate folder
        if passed:
            shutil.copy(mask_path, os.path.join(passed_folder, mask_file))
        else:
            shutil.copy(mask_path, os.path.join(failed_folder, mask_file))
    
    print(f"Filtering complete. Results saved in {output_folder}")
    print(f"Passed masks: {passed_folder}")
    print(f"Failed masks: {failed_folder}")

def main():
    parser = argparse.ArgumentParser(description="Filter mask images by size and/or number of connected components.")
    parser.add_argument("input_folder", help="Path to the folder containing mask images")
    parser.add_argument("output_folder", help="Path to save the filtered mask images")
    parser.add_argument("--min_size_kb", type=int, 
                       help="Minimum file size in KB for a valid mask")
    parser.add_argument("--max_components", type=int, 
                       help="Maximum allowed connected components")
    
    args = parser.parse_args()
    
    if args.min_size_kb is None and args.max_components is None:
        parser.error("At least one filtering criterion (--min_size_kb or --max_components) must be provided")
    
    filter_masks(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        min_size_kb=args.min_size_kb,
        max_components=args.max_components
    )

if __name__ == "__main__":
    main()