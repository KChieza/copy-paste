import os
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from ultralytics import SAM
from tqdm import tqdm
import warnings

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def filter_non_occluded_and_non_truncated_bboxes(bboxes, centers, image_width, image_height, iou_threshold=0.1):
    filtered_bboxes = []
    filtered_centers = []
    
    for i, box in enumerate(bboxes):
        x_min, y_min, x_max, y_max = box
        occluded = False
        truncated = (x_min <= 0 or y_min <= 0 or x_max >= image_width or y_max >= image_height)
        
        for j, other_box in enumerate(bboxes):
            if i != j and compute_iou(box, other_box) > iou_threshold:
                occluded = True
                break
        
        if not occluded and not truncated:
            filtered_bboxes.append(box)
            filtered_centers.append(centers[i])

    return filtered_bboxes, filtered_centers

def yolo_to_absolute_prompts(label_path, image_width, image_height):
    bboxes = []
    centers = []
    with open(label_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, x_center, y_center, width, height = map(float, parts)
            x_center *= image_width
            y_center *= image_height
            width *= image_width
            height *= image_height
            x_min = x_center - width / 2
            y_min = y_center - height / 2
            x_max = x_center + width / 2
            y_max = y_center + height / 2
            bboxes.append([x_min, y_min, x_max, y_max])
            centers.append([x_center, y_center])
    return bboxes, centers

def run_sam_with_yolo(image_path, label_path, model, output_folder, iou_threshold=0.1):
    image = Image.open(image_path)
    image_width, image_height = image.size
    
    bboxes, centers = yolo_to_absolute_prompts(label_path, image_width, image_height)
    
    filtered_bboxes, filtered_centers = filter_non_occluded_and_non_truncated_bboxes(
        bboxes, centers, image_width, image_height, iou_threshold)
    
    if not filtered_bboxes:
        return 0
    
    # Suppress the default output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = model(image_path, bboxes=filtered_bboxes, points=filtered_centers, verbose=False)
    
    image_array = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2BGRA)
    
    mask_count = 0
    
    for obj_idx, c in enumerate(results):
        for mask_idx, mask in enumerate(c.masks.xy):
            contour = mask.astype(np.int32).reshape(-1, 1, 2)
            transparent_image = image_array.copy()
            
            mask_image = np.zeros(image_array.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask_image, [contour], -1, 255, thickness=cv2.FILLED)
            transparent_image[mask_image == 0] = [0, 0, 0, 0]
            
            x, y, w, h = cv2.boundingRect(contour)
            cropped_image = transparent_image[y:y+h, x:x+w]
            
            label = c.names[c.boxes.cls.tolist()[obj_idx]]
            output_file = os.path.join(output_folder, f"{Path(image_path).stem}_{label}_{obj_idx}_{mask_idx}.png")
            cv2.imwrite(output_file, cropped_image)
            mask_count += 1
    
    return mask_count

def process_folder(images_folder, labels_folder, output_folder, model_path="sam2.1_b.pt", iou_threshold=0.1):
    os.makedirs(output_folder, exist_ok=True)
    model = SAM(model_path)
    
    image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_masks = 0
    
    print(f"Processing {len(image_files)} images...")
    with tqdm(image_files, desc="Extracting masks") as pbar:
        for image_file in pbar:
            image_path = os.path.join(images_folder, image_file)
            label_path = os.path.join(labels_folder, f"{Path(image_file).stem}.txt")
            
            if not os.path.exists(label_path):
                pbar.write(f"Label file not found for {image_file}, skipping.")
                continue
            
            masks_extracted = run_sam_with_yolo(image_path, label_path, model, output_folder, iou_threshold)
            total_masks += masks_extracted
            pbar.set_postfix({"Masks": total_masks})
    
    print(f"\nProcessing complete!")
    print(f"Total masks extracted: {total_masks}")
    print(f"Masks saved to: {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAM on YOLO-annotated dataset")
    parser.add_argument('--images', type=str, required=True, help="Path to folder containing images")
    parser.add_argument('--labels', type=str, required=True, help="Path to folder containing YOLO labels")
    parser.add_argument('--output', type=str, required=True, help="Output folder to save results")
    parser.add_argument('--model', type=str, default="sam2.1_b.pt", help="Path to SAM model weights")
    parser.add_argument('--iou-threshold', type=float, default=0.1, help="IoU threshold for occlusion filtering")

    args = parser.parse_args()

    process_folder(args.images, args.labels, args.output, args.model, args.iou_threshold)