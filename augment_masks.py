import cv2
import os
import random
import numpy as np
from pathlib import Path
import argparse


def rotate_image(image, angle):
    """
    Rotate the image by a specific angle.
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return rotated


def flip_image_horizontally(image):
    """
    Flip the image horizontally.
    """
    return cv2.flip(image, 1)


def scale_image(image, scale_factor):
    """
    Scale the image by a factor.
    """
    (h, w) = image.shape[:2]
    scaled_w = int(w * scale_factor)
    scaled_h = int(h * scale_factor)

    scaled = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
    return scaled


def apply_augmentations_to_masks(masks_folder, augmented_masks_folder, num_augmented_versions=5,
                                rotation_range=(-30, 30), scale_range=(0.7, 1.3)):
    """
    Apply augmentations to masks and save augmented versions.
    """
    os.makedirs(augmented_masks_folder, exist_ok=True)

    mask_files = [f for f in os.listdir(masks_folder) if f.lower().endswith('.png')]

    for mask_file in mask_files:
        mask_path = os.path.join(masks_folder, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if mask is None or mask.shape[2] != 4:  # Ensure it's a valid mask with alpha channel
            print(f"Skipping invalid mask: {mask_file}")
            continue

        for i in range(num_augmented_versions):
            augmented_mask = mask.copy()

            # Randomly apply valid augmentations
            if random.choice([True, False]):
                angle = random.uniform(rotation_range[0], rotation_range[1])
                augmented_mask = rotate_image(augmented_mask, angle)

            if random.choice([True, False]):
                augmented_mask = flip_image_horizontally(augmented_mask)

            if random.choice([True, False]):
                scale_factor = random.uniform(scale_range[0], scale_range[1])
                augmented_mask = scale_image(augmented_mask, scale_factor)

            # Save the augmented mask
            augmented_mask_filename = f"{Path(mask_file).stem}_aug_{i}.png"
            augmented_mask_path = os.path.join(augmented_masks_folder, augmented_mask_filename)
            cv2.imwrite(augmented_mask_path, augmented_mask)

    print("Augmented masks saved to:", augmented_masks_folder)


def parse_range_argument(arg):
    """Helper function to parse min,max range arguments"""
    try:
        min_val, max_val = map(float, arg.split(','))
        return (min_val, max_val)
    except:
        raise argparse.ArgumentTypeError("Range must be in format 'min,max' (e.g., '-30,30')")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Apply augmentations to mask images.')
    
    # Required arguments
    parser.add_argument('input_folder', type=str, help='Path to the folder containing original masks')
    parser.add_argument('output_folder', type=str, help='Path to save the augmented masks')
    
    # Optional arguments
    parser.add_argument('--num_augmentations', type=int, default=5,
                       help='Number of augmented versions to create per mask (default: 5)')
    parser.add_argument('--rotation_range', type=parse_range_argument, default='-30,30',
                       help='Rotation angle range in degrees (min,max) (default: -30,30)')
    parser.add_argument('--scale_range', type=parse_range_argument, default='0.7,1.3',
                       help='Scaling factor range (min,max) (default: 0.7,1.3)')
    
    # Parse arguments
    args = parser.parse_args()
    

    # Call the function with command-line arguments
    apply_augmentations_to_masks(
        masks_folder=args.input_folder,
        augmented_masks_folder=args.output_folder,
        num_augmented_versions=args.num_augmentations,
        rotation_range=args.rotation_range,
        scale_range=args.scale_range
    )


if __name__ == "__main__":
    main()