import os
import csv
from ultralytics import YOLO

# Define models and directories
models = [
    "yolov9e.pt",
    "yolov10x.pt",
    "yolo11x.pt",
    "yolo12x.pt"
]

# Define dataset and training parameters
project_name = "sey_beriv3_baselines"
dataset = "sey_beri_v3"
data_yaml = f"{dataset}/data.yaml"  # Path to the data.yaml file
epochs = 300
training_params = dict(
    data=data_yaml,
    epochs=epochs,
    patience = 50,
    close_mosaic = 0,
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    degrees=0, translate=0.0, scale=0.0,
    shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.0,
    mosaic=False, mixup=False, augment=False,
    erasing=0.0, crop_fraction=0.0, plots = True, project = project_name
)
# training_params = dict(
#     data=data_yaml,
#     epochs=epochs,
#     hsv_h = 0.0, hsv_s = 0.0, hsv_v = 0.0, 
#     degrees = 0, translate = 0.1, scale = 0.5, 
#     shear = 0.0, perspective = 0.0, flipud = 0.0, fliplr = 0.5, 
#     mosaic = True, mixup = False, 
#     erasing=0.4, crop_fraction = 0.0
# )

# Define validation parameters
val_params = dict(
    split="test",
    imgsz=640,
    conf=0.5,
    iou=0.6,
    project = project_name,
    plots = True
)

# CSV file for results
csv_file = "runs/detect/test_results.csv"

# Training & Validation Loop
for model_name in models:
    print(f"\n🚀 Training {model_name}...\n")
    
    # Initialize and train model
    model = YOLO(model_name)
    run_name = f"train_{model_name.split('.')[0]}_{dataset}_optaug"
    training_params["name"] = run_name
    model.train(**training_params)

    # Load best model after training
    best_model_path = f"runs/detect/{run_name}/weights/best.pt"
    if not os.path.exists(best_model_path):
        print(f"❌ Best model not found for {model_name}, skipping validation.")
        continue

    print(f"\n📊 Validating {model_name}...\n")

    # Load trained model and validate
    model = YOLO(best_model_path)
    val_name = f"test_{model_name.split('.')[0]}_{dataset}_optaug"
    val_params["name"] = val_name
    metrics = model.val(**val_params)

    # Extract and format results
    row_data = {
        "run": val_name,
        "conf": val_params["conf"],
        "iou": val_params["iou"],
        "imgsz": val_params["imgsz"],
        "precision": f"{metrics.box.p[0] * 100:.2f}",
        "recall": f"{metrics.box.r[0] * 100:.2f}",
        "f1_score": f"{metrics.box.f1[0] * 100:.2f}",
        "map50": f"{metrics.box.map50 * 100:.2f}",
        "map75": f"{metrics.box.map75 * 100:.2f}",
        "map50_95": f"{metrics.box.map * 100:.2f}"
    }

    # Write results to CSV
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()  # Write headers if file is new
        writer.writerow(row_data)

    print(f"✅ Validation results saved for {model_name}.\n")

print("🎯 All models trained and evaluated!")
