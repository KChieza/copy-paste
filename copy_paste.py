import shutil
import os
import random
import cv2
import numpy as np
from pathlib import Path
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Augment images by pasting random objects from masks.')
    parser.add_argument('--images', required=True, help='Path to folder containing original images')
    parser.add_argument('--labels', required=True, help='Path to folder containing original labels')
    parser.add_argument('--masks', required=True, help='Path to folder containing mask images')
    parser.add_argument('--output', required=True, help='Path to output folder for saving augmented dataset')
    parser.add_argument('--prob', type=float, required=True, 
                       help='Probability of augmenting each image (0.0 to 1.0)')
    parser.add_argument('--min-objects', type=int, default=5, 
                       help='Minimum number of objects to paste (default: 5)')
    parser.add_argument('--max-objects', type=int, default=10, 
                       help='Maximum number of objects to paste (default: 10)')
    parser.add_argument('--replace', action='store_true', 
                       help='Overwrite images selected for augmentation with their augmented versions instead of creating new ones')
    parser.add_argument('--env-paste', action='store_true', 
                       help='Only paste masks from the same environment')
    return parser.parse_args()

def get_environment(filename):
    """Extracts the environment name from the filename."""
    for env in ["EnvironmentA", "EnvironmentB", "EnvironmentC", "EnvironmentD", "EnvironmentE"]:
        if filename.startswith(env):
            return env
    return None

def get_non_intersecting_bbox(image_shape, existing_bboxes, obj_width, obj_height, max_attempts=50):
    """Find a non-intersecting bounding box for placing an object."""
    image_height, image_width = image_shape
    for _ in range(max_attempts):
        x1 = random.randint(0, image_width - obj_width)
        y1 = random.randint(0, image_height - obj_height)
        x2, y2 = x1 + obj_width, y1 + obj_height
        new_bbox = [x1, y1, x2, y2]
        if all(not is_intersecting(new_bbox, bbox) for bbox in existing_bboxes):
            return new_bbox
    return None

def is_intersecting(bbox1, bbox2):
    """Check if two bounding boxes intersect."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    return not (x2_1 <= x1_2 or x2_2 <= x1_1 or y2_1 <= y1_2 or y2_2 <= y1_1)

def convert_to_yolo_format(bbox, image_width, image_height, class_id):
    """Convert bounding box to YOLO format."""
    x1, y1, x2, y2 = bbox
    x_center = ((x1 + x2) / 2) / image_width
    y_center = ((y1 + y2) / 2) / image_height
    bbox_width = (x2 - x1) / image_width
    bbox_height = (y2 - y1) / image_height
    return f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"


def copy_original_files(input_images, input_labels, output_images, output_labels):
    """
    Copy original images and labels to output folders without overwriting existing files.
    """
    # Copy images
    for file_name in os.listdir(input_images):
        src_path = os.path.join(input_images, file_name)
        dest_path = os.path.join(output_images, file_name)
        
        if not os.path.exists(dest_path):
            shutil.copy2(src_path, dest_path)

    # Copy labels
    for file_name in os.listdir(input_labels):
        src_path = os.path.join(input_labels, file_name)
        dest_path = os.path.join(output_labels, f"{Path(file_name).stem}.txt")
        
        if not os.path.exists(dest_path):
            shutil.copy2(src_path, dest_path)

def augment_images_with_labels(args):
    """
    Augment images by pasting random objects from the masks folder with a given probability.
    Always copies non-augmented images to maintain complete dataset in output.
    """
    output_images_folder = os.path.join(args.output, "images")
    output_labels_folder = os.path.join(args.output, "labels")
    os.makedirs(output_images_folder, exist_ok=True)
    os.makedirs(output_labels_folder, exist_ok=True)
    
    # First copy all original files to output (will be overwritten if --replace flag used)
    copy_original_files(args.images, args.labels, output_images_folder, output_labels_folder)
    
    image_files = [f for f in os.listdir(args.images) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    mask_files = [f for f in os.listdir(args.masks) if f.lower().endswith('.png')]
    
    # Organize masks by environment
    env_mask_dict = {}
    for mask_file in mask_files:
        env = get_environment(mask_file)
        if env:
            env_mask_dict.setdefault(env, []).append(mask_file)
    
    augmented_count = 0
    
    for image_file in image_files:
        if random.random() > args.prob:
            continue  # Skip this image (already copied original)
        
        image_path = os.path.join(args.images, image_file)
        label_path = os.path.join(args.labels, f"{Path(image_file).stem}.txt")
        
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Failed to load image: {image_path}, skipping.")
            continue
        
        image_height, image_width = image.shape[:2]
        existing_bboxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    try:
                        class_id, x_center, y_center, bbox_width, bbox_height = map(float, line.strip().split())
                        x1 = int((x_center - bbox_width / 2) * image_width)
                        y1 = int((y_center - bbox_height / 2) * image_height)
                        x2 = int((x_center + bbox_width / 2) * image_width)
                        y2 = int((y_center + bbox_height / 2) * image_height)
                        existing_bboxes.append([x1, y1, x2, y2])
                    except ValueError:
                        print(f"Error reading label file: {label_path}, skipping.")
                        continue
        
        new_image_name = image_file if args.replace else f"{Path(image_file).stem}_aug{Path(image_file).suffix}"
        new_image_path = os.path.join(output_images_folder, new_image_name)
        new_label_name = f"{Path(image_file).stem}.txt" if args.replace else f"{Path(image_file).stem}_aug.txt"
        new_label_path = os.path.join(output_labels_folder, new_label_name)

        # Copy original labels to new label file (if not replacing)
        if not args.replace and os.path.exists(label_path):
            with open(label_path, "r") as original_label_file:
                with open(new_label_path, "w") as new_label_file:
                    if os.path.getsize(label_path) > 0:
                        new_label_file.writelines(original_label_file.readlines())
                        new_label_file.write("\n")

        with open(new_label_path, "a") as label_file:
            if args.replace and os.path.getsize(new_label_path) > 0:
                label_file.write("\n")
            env = get_environment(image_file)
            available_masks = env_mask_dict.get(env, []) if args.env_paste else mask_files
            
            # Determine number of objects to paste based on the range
            num_objects = random.randint(args.min_objects, args.max_objects)
            sampled_masks = random.sample(available_masks, min(num_objects, len(available_masks)))
            
            for mask_file in sampled_masks:
                mask_path = os.path.join(args.masks, mask_file)
                mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                if mask is None or mask.shape[2] != 4:
                    continue
                
                obj_height, obj_width = mask.shape[:2]
                new_bbox = get_non_intersecting_bbox((image_height, image_width), existing_bboxes, obj_width, obj_height)
                if new_bbox is None:
                    continue
                
                x1, y1, x2, y2 = new_bbox
                existing_bboxes.append(new_bbox)
                for c in range(3):
                    image[y1:y2, x1:x2, c] = np.where(mask[:, :, 3] > 0, mask[:, :, c], image[y1:y2, x1:x2, c])
                
                label_file.write(convert_to_yolo_format(new_bbox, image_width, image_height, 0) + "\n")
            
            cv2.imwrite(new_image_path, image)

            if args.replace:
                # Add '_aug' suffix to both image and label files
                base_name = Path(image_file).stem
                new_aug_image_name = f"{base_name}_aug{Path(image_file).suffix}"
                new_aug_label_name = f"{base_name}_aug.txt"
                
                # Rename image file
                aug_image_path = os.path.join(output_images_folder, new_aug_image_name)
                os.rename(new_image_path, aug_image_path)
                
                # Rename label file
                aug_label_path = os.path.join(output_labels_folder, new_aug_label_name)
                os.rename(new_label_path, aug_label_path)
            augmented_count += 1
    
    print(f"Total augmented images: {augmented_count}")
    print(f"Complete dataset saved to: {args.output}")

def main():
    args = parse_arguments()
    
    # Validate probability
    if not 0 <= args.prob <= 1:
        print("Error: Probability must be between 0 and 1")
        return
    
    # Validate object count range
    if args.min_objects < 0 or args.max_objects < args.min_objects:
        print("Error: Invalid object count range")
        return
    
    augment_images_with_labels(args)
    print("Done")

if __name__ == "__main__":
    main()
