import shutil
import os
import random
import cv2
import numpy as np
from pathlib import Path

def get_environment(filename):
    """Extracts the environment name from the filename."""
    for env in ["EnvironmentA", "EnvironmentB", "EnvironmentC", "EnvironmentD", "frame"]:
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

def augment_images_with_labels(images_folder, labels_folder, masks_folder, output_folder, 
                             augmentation_prob, obj_count_range=(5, 10), 
                             replace_existing=False, use_env_pasting=True):
    """
    Augment images by pasting random objects from the masks folder with a given probability.
    
    Args:
        images_folder (str): Path to the folder containing original images.
        labels_folder (str): Path to the folder containing original labels.
        masks_folder (str): Path to the folder containing masks.
        output_folder (str): Path to save augmented images and labels.
        augmentation_prob (float): Probability of applying augmentation to an image (0 to 1).
        obj_count_range (tuple): Range of objects to paste (min, max).
        replace_existing (bool): Whether to replace existing images or create new ones.
        use_env_pasting (bool): Whether to paste masks only within the same environment.
    """
    output_images_folder = os.path.join(output_folder, "images")
    output_labels_folder = os.path.join(output_folder, "labels")
    os.makedirs(output_images_folder, exist_ok=True)
    os.makedirs(output_labels_folder, exist_ok=True)
    
    image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    mask_files = [f for f in os.listdir(masks_folder) if f.lower().endswith('.png')]
    
    # Organize masks by environment
    env_mask_dict = {}
    for mask_file in mask_files:
        env = get_environment(mask_file)
        if env:
            env_mask_dict.setdefault(env, []).append(mask_file)
    
    augmented_count = 0
    
    for image_file in image_files:
        image_path = os.path.join(images_folder, image_file)
        label_path = os.path.join(labels_folder, f"{Path(image_file).stem}.txt")
        
        if random.random() > augmentation_prob:
            continue
        
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
        
        new_image_name = image_file if replace_existing else f"{Path(image_file).stem}_aug{Path(image_file).suffix}"
        new_image_path = os.path.join(output_images_folder, new_image_name)
        new_label_name = f"{Path(image_file).stem}.txt" if replace_existing else f"{Path(image_file).stem}_aug.txt"
        new_label_path = os.path.join(output_labels_folder, new_label_name)

        # Copy original labels to new label file (if not replacing)
        if not replace_existing and os.path.exists(label_path):
            with open(label_path, "r") as original_label_file:
                with open(new_label_path, "w") as new_label_file:
                    if os.path.getsize(label_path) > 0:
                        new_label_file.writelines(original_label_file.readlines())
                        new_label_file.write("\n")

        with open(new_label_path, "a" if replace_existing else "a") as label_file:
            env = get_environment(image_file)
            available_masks = env_mask_dict.get(env, []) if use_env_pasting else mask_files
            
            # Determine number of objects to paste based on the range
            min_objs, max_objs = obj_count_range
            num_objects = random.randint(min_objs, max_objs)
            sampled_masks = random.sample(available_masks, min(num_objects, len(available_masks)))
            
            for mask_file in sampled_masks:
                mask_path = os.path.join(masks_folder, mask_file)
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
            augmented_count += 1
    
    print(f"Total augmented images: {augmented_count}")

def copy_files_without_overwrite(input_images, input_labels, output_folder):
    """
    Copy images and labels from input folders to output folders without overwriting existing files.
    """
    output_images = f"{output_folder}/images"
    output_labels = f"{output_folder}/labels"
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_labels, exist_ok=True)

    # Copy images
    for file_name in os.listdir(input_images):
        src_path = os.path.join(input_images, file_name)
        dest_path = os.path.join(output_images, file_name)
        
        if not os.path.exists(dest_path):
            shutil.copy2(src_path, dest_path)

    # Copy labels
    for file_name in os.listdir(input_labels):
        src_path = os.path.join(input_labels, file_name)
        dest_path = os.path.join(output_labels, file_name)
        
        if not os.path.exists(dest_path):
            shutil.copy2(src_path, dest_path)

# Example usage
probs = [0.3,0.5,0.7]
obj_count_range = (5, 10)  # Now you can specify any range you want
replace_existing = False
use_env_pasting = False
images_folder = "../sey_beri_v3/train/images"
labels_folder = "../sey_beri_v3/train/labels"
masks_folder = "../sey_beri_v3/masks/passed_aug"

for augmentation_prob in probs:
    output_folder = f"../sey_beri_v3/sey_beri_v3_randpaste_augmask_{int(augmentation_prob * 100)}prob_rp{replace_existing}"
    augment_images_with_labels(
        images_folder=images_folder,
        labels_folder=labels_folder,
        masks_folder=masks_folder,
        output_folder=output_folder,
        augmentation_prob=augmentation_prob,
        obj_count_range=obj_count_range,
        replace_existing=replace_existing,
        use_env_pasting=use_env_pasting
    )
    copy_files_without_overwrite(images_folder, labels_folder, output_folder)

print("Done")



