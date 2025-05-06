# Copy-Paste Augmentation for Fish Detection

## Overview

The effectiveness of fish detection models is heavily influenced by the quantity and diversity of training data available. In this study, we investigate the use of **copy-paste augmentation**, a simple yet effective data augmentation technique, to enhance model performance by synthetically increasing dataset size and variability.

This approach involves copying fish instances from existing training images and pasting them onto other images within the dataset, creating more diverse training scenes. Through a series of experiments, we evaluate the impact of copy-paste on model performance, comparing it to baseline training without augmentation.

> Our findings indicate that copy-paste improves model precision and localization accuracy, demonstrating its value especially when working with small datasets.

---

## Requirements

Before using this project, install the required packages:

```bash
pip install -r requirements.txt
```

---

## Dataset Format

Ensure your training dataset follows the YOLOv5 format:

```
train/
├── images/
│   ├── image1.jpg
│   └── ...
└── labels/
    ├── image1.txt
    └── ...
```

Each label file should contain one or more annotations in YOLO format:

```
<class_id> <x_center> <y_center> <width> <height>
```

---

## 1. Extracting Fish Masks

The first step is to extract clean fish instances (masks) using a SAM (Segment Anything Model) guided by YOLO bounding boxes. We filter out truncated and occluded fish to ensure high-quality paste targets.

### Usage

```bash
python get_masks.py \
  --images path/to/train/images \
  --labels path/to/train/labels \
  --output path/to/output/masks \
  --model sam2.1_b.pt \
  --iou-threshold 0.1
```

### Command Line Arguments

* `--images`: Path to the directory containing training images.
* `--labels`: Path to the corresponding YOLO-format label files.
* `--output`: Directory where the extracted masks will be saved.
* `--model`: Path to the SAM2 model checkpoint (see model options below).
* `--iou-threshold`: *(Optional)* IoU threshold for occlusion filtering. Defaults to `0.1`. Fish whose bounding boxes overlap with others by more than this threshold will be skipped, as occlusion often leads to poor quality masks.

### 🧠 Supported SAM2 Models

You can use any of the following SAM2 or SAM2.1 models with this script:

| Model Name   | Checkpoint    |
| ------------ | ------------- |
| SAM 2 Tiny   | `sam2_t.pt`   |
| SAM 2 Small  | `sam2_s.pt`   |
| SAM 2 Base   | `sam2_b.pt`   |
| SAM 2 Large  | `sam2_l.pt`   |

If the specified model weights are not found locally, they will be automatically downloaded to your current working directory.

📘 **More info:** [Ultralytics SAM2 documentation](https://docs.ultralytics.com/models/sam-2/#how-to-use-sam-2-versatility-in-image-and-video-segmentation)

### 📦 Example Output

Extracted fish instances will be saved as transparent PNGs in the specified output directory:

```
output/masks/
├── frame_00001_fish_0_0.png
├── frame_00001_fish_1_0.png
└── ...
```
---

## 2. Filtering

This step filters out poor-quality masks either **automatically** based on size or structure, or **manually** by reviewing masks one-by-one.

### 2.1 Automatic Filtering

**Script:** auto\_filtering.py
This script filters masks based on:

* Minimum file size in kilobytes (--min\_size\_kb)
* Maximum number of connected components (--max\_components)

Masks are sorted into:

* output\_folder/passed/
* output\_folder/failed/

**Usage:**

```bash
python auto_filtering.py <input_folder> <output_folder> --min_size_kb 10 --max_components 3
```

You must specify at least one of --min\_size\_kb or --max\_components.

---

### 2.2 Manual Filtering

**Script:** manual\_filtering.py
This interactive script allows visual inspection of masks, sorting them with keyboard controls:

* Press y to keep an image.
* Press n to move it to the failed folder.
* Press u to undo the last action.
* Press q to quit.

The tool also tracks progress across sessions using a progress file.

**Usage:**

```bash
python manual_filtering.py --input path/to/masks --failed path/to/failed --progress progress.txt
```

Optional display settings:

```bash
--screen_width 1920 --screen_height 1080
```

---

## 3. Augmenting Masks

This step expands the variety of fish masks by applying random augmentations including rotation, horizontal flipping, and scaling.

**Script:** augment\_masks.py

**Features:**

* Rotation within a specified angle range
* Horizontal flipping
* Scaling by a random factor

**Usage:**

```bash
python augment_masks.py input_folder output_folder \
  --num_augmentations 5 \
  --rotation_range -30,30 \
  --scale_range 0.7,1.3
```

* `input_folder`: Path to folder containing original masks
* `output_folder`: Where to save augmented masks
* `--num_augmentations`: Number of augmented versions to create per original mask (default: 5)
* `--rotation_range`: Range of rotation angles in degrees (default: -30,30)
* `--scale_range`: Range of scaling factors (default: 0.7,1.3)

Each augmented mask is saved as a transparent PNG with the suffix `_aug_<i>.png`.

---

## 4. Copy-Paste Augmentation

This script augments an image dataset by pasting segmented fish masks into original images to increase dataset diversity and improve training performance. It supports pasting from the same environment only (optional), and outputs YOLO-format labels for the newly added objects.

### Script: `copy_paste.py`

**Features:**

* Applies copy-paste augmentation to selected images with a user-defined probability
* Supports choosing a random number of pasted objects within a specified range
* Allows optional environment-constrained pasting
* Automatically avoids overlapping pasted masks
* Keeps all original data in the output folder and can optionally overwrite selected images

### Usage:

```bash
python copy_paste.py \
  --images path/to/images \
  --labels path/to/labels \
  --masks path/to/masks \
  --output path/to/output \
  --prob 0.5 \
  --min-objects 5 \
  --max-objects 10 \
  [--replace] \
  [--env-paste]
```

### Parameters:

* `--images`: Path to folder with original images
* `--labels`: Path to folder with original YOLO-format labels
* `--masks`: Folder of PNG masks with alpha channel
* `--output`: Where to save the augmented dataset
* `--prob`: Probability (0–1) of applying copy-paste to each image
* `--min-objects`: Minimum number of masks to paste
* `--max-objects`: Maximum number of masks to paste
* `--replace`: (Optional) Overwrite original image with augmented version (otherwise creates new files with `_aug` suffix)
* `--env-paste`: (Optional) Only paste masks from the same environment as the image (e.g., EnvironmentA)

**Note:**

* For environment-aware pasting to work, your image filenames must begin with the environment name.
* The default environment names are:

  ```python
  ["EnvironmentA", "EnvironmentB", "EnvironmentC", "EnvironmentD"]
  ```
* If your filenames follow a different naming convention, you can modify the `get_environment` function in the script to suit your needs.

### Output:

* All original and augmented images and labels are saved to:

  * `output/images`
  * `output/labels`
* New files are suffixed with `_aug` unless `--replace` is used

### Example:

```bash
python copy_paste.py \
  --images data/images \
  --labels data/labels \
  --masks clean_masks \
  --output augmented_output \
  --prob 0.4 \
  --min-objects 3 \
  --max-objects 6 \
  --env-paste
```

This command augments 40% of the images, pasting 3–6 objects from the same environment, and stores results in `augmented_output`.

---
