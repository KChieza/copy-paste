import os
import yaml
import shutil
import csv
from ultralytics import YOLO

# Define model, dataset directory, and parameters
project_name = "runs/sey_beriv3_cp"
model_name = "yolov9e.pt"  # Change this to your desired model
dataset_folder = "sey_beri_v3"
data_yaml_path = os.path.join(dataset_folder, "data.yaml")
epochs = 300

# Backup original data.yaml
backup_yaml_path = data_yaml_path + ".bak"
shutil.copy(data_yaml_path, backup_yaml_path)

# Load original data.yaml
with open(data_yaml_path, "r") as file:
    data_yaml = yaml.safe_load(file)

# Define the list of datasets to be included
included_folders = {"train_bgyolo"}

dataset_variants = [
    d for d in os.listdir(dataset_folder)
    if os.path.isdir(os.path.join(dataset_folder, d)) and d in included_folders
]

# CSV file for results
csv_file = f"{project_name}/test_results.csv"
os.makedirs(os.path.dirname(csv_file), exist_ok=True)

# Train and validate on each dataset variant
for dataset_variant in dataset_variants:
    print(f"\n🚀 Training on {dataset_variant}...\n")

    # Update data.yaml for the current dataset variant
    data_yaml["train"] = f"../{dataset_variant}/images"
    with open(data_yaml_path, "w") as file:
        yaml.dump(data_yaml, file)

    # Train model
    model = YOLO(model_name)
    run_name = f"train_{model_name.split('.')[0]}_{dataset_variant}_optaug"

    training_params = dict(
        data=data_yaml_path,
        epochs=epochs, patience=50,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, 
        degrees=0, translate=0.1, scale=0.5, 
        shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, 
        mosaic=True, mixup=False, 
        erasing=0.4, crop_fraction=0.0,
        name=run_name, plots=True, project=project_name
    )

    model.train(**training_params)

    # Load best model for validation
    best_model_path = f"{project_name}/{run_name}/weights/best.pt"
    if not os.path.exists(best_model_path):
        print(f"❌ Best model not found for {dataset_variant}, skipping validation.")
        continue

    print(f"\n📊 Validating on {dataset_variant}...\n")

    # Validate model
    model = YOLO(best_model_path)
    metrics = model.val(split="test", imgsz=640, conf=0.5, iou=0.6, name=f"test_{model_name.split('.')[0]}_{dataset_variant}_optaug", plots=True, project=project_name)

    # Extract and format results
    row_data = {
        "dataset": dataset_variant,
        "precision": f"{metrics.box.p[0] * 100:.2f}",
        "recall": f"{metrics.box.r[0] * 100:.2f}",
        "f1_score": f"{metrics.box.f1[0] * 100:.2f}",
        "map50": f"{metrics.box.map50 * 100:.2f}",
        "map75": f"{metrics.box.map75 * 100:.2f}",
        "map50_95": f"{metrics.box.map * 100:.2f}"
    }

    # Write results to CSV
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

    print(f"✅ Results saved for {dataset_variant}.\n")

# Restore original data.yaml
shutil.move(backup_yaml_path, data_yaml_path)
print("🎯 All datasets processed! data.yaml restored.")

exit()
import os
import yaml
import shutil
import csv
from ultralytics import YOLO

# Define model, dataset directory, and parameters
project_name = "runs/sey_beriv3_cp"
model_name = "yolov9e.pt"  # Change this to your desired model
dataset_folder = "sey_beri_v3"
data_yaml_path = os.path.join(dataset_folder, "data.yaml")
epochs = 300

# Backup original data.yaml
backup_yaml_path = data_yaml_path + ".bak"
shutil.copy(data_yaml_path, backup_yaml_path)

# Load original data.yaml
with open(data_yaml_path, "r") as file:
    data_yaml = yaml.safe_load(file)

# Identify dataset variants (excluding train, val, and test)
excluded_folders = {"val", "test", "masks", "all_data", "train", "sey_beri_v3_augmask_70prob_rpTrue", "sey_beri_v3_augmask_50prob_rpFalse", "sey_beri_v3_envpaste_augmask_50prob_rpFalse", "sey_beri_v3_envpaste_augmask_70prob_rpFalse"}
dataset_variants = [
    d for d in os.listdir(dataset_folder)
    if os.path.isdir(os.path.join(dataset_folder, d)) and d not in excluded_folders
]

# CSV file for results
csv_file = f"{project_name}/test_results.csv"
os.makedirs(os.path.dirname(csv_file), exist_ok=True)

# Train and validate on each dataset variant
for dataset_variant in dataset_variants:
    print(f"\n🚀 Training on {dataset_variant}...\n")

    # Update data.yaml for the current dataset variant
    data_yaml["train"] = f"../{dataset_variant}/images"
    with open(data_yaml_path, "w") as file:
        yaml.dump(data_yaml, file)

    # Train model
    model = YOLO(model_name)
    run_name = f"train_{model_name.split('.')[0]}_{dataset_variant}_optaug"

    # training_params = dict(
    # data=data_yaml_path,
    # epochs=epochs, patience = 50,
    # close_mosaic = 0,
    # hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    # degrees=0, translate=0.0, scale=0.0,
    # shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.0,
    # mosaic=False, mixup=False, augment=False,
    # erasing=0.0, crop_fraction=0.0, name=run_name,
    # plots = True, project = project_name)

    training_params = dict(
    data=data_yaml_path,
    epochs=epochs, patience = 50,
    hsv_h = 0.0, hsv_s = 0.0, hsv_v = 0.0, 
    degrees = 0, translate = 0.1, scale = 0.5, 
    shear = 0.0, perspective = 0.0, flipud = 0.0, fliplr = 0.5, 
    mosaic = True, mixup = False, 
    erasing=0.4, crop_fraction = 0.0,
    name=run_name, plots = True, project = project_name)

    model.train(**training_params)

    # Load best model for validation
    best_model_path = f"{project_name}/{run_name}/weights/best.pt"
    if not os.path.exists(best_model_path):
        print(f"❌ Best model not found for {dataset_variant}, skipping validation.")
        continue

    print(f"\n📊 Validating on {dataset_variant}...\n")

    # Validate model
    model = YOLO(best_model_path)
    metrics = model.val(split="test", imgsz=640, conf=0.5, iou=0.6, name=f"test_{model_name.split('.')[0]}_{dataset_variant}_optaug", plots = True, project = project_name)

    # Extract and format results
    row_data = {
        "dataset": dataset_variant,
        "precision": f"{metrics.box.p[0] * 100:.2f}",
        "recall": f"{metrics.box.r[0] * 100:.2f}",
        "f1_score": f"{metrics.box.f1[0] * 100:.2f}",
        "map50": f"{metrics.box.map50 * 100:.2f}",
        "map75": f"{metrics.box.map75 * 100:.2f}",
        "map50_95": f"{metrics.box.map * 100:.2f}"
    }

    # Write results to CSV
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

    print(f"✅ Results saved for {dataset_variant}.\n")

# Restore original data.yaml
shutil.move(backup_yaml_path, data_yaml_path)
print("🎯 All datasets processed! data.yaml restored.")
