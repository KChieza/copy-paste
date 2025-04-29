import os
import shutil

def copy_files_without_overwrite(input_images, input_labels, output_images, output_labels):
    """
    Copy images and labels from input folders to output folders without overwriting existing files.
    
    Args:
        input_images (str): Path to the input images folder.
        input_labels (str): Path to the input labels folder.
        output_images (str): Path to the output images folder.
        output_labels (str): Path to the output labels folder.
    """
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_labels, exist_ok=True)

    # Copy images
    for file_name in os.listdir(input_images):
        src_path = os.path.join(input_images, file_name)
        dest_path = os.path.join(output_images, file_name)
        
        if not os.path.exists(dest_path):  # Only copy if the file does not exist
            shutil.copy2(src_path, dest_path)
        else:
            print(f"Skipping existing image: {file_name}")

    # Copy labels
    for file_name in os.listdir(input_labels):
        src_path = os.path.join(input_labels, file_name)
        dest_path = os.path.join(output_labels, file_name)
        
        if not os.path.exists(dest_path):  # Only copy if the file does not exist
            shutil.copy2(src_path, dest_path)
        else:
            print(f"Skipping existing label: {file_name}")

# Example usage
input_images = "deepfish/train/images"
input_labels = "deepfish/train/labels"
output_images = "deepfish/deepfish_30prob/images"
output_labels = "deepfish/deepfish_30prob/labels"

copy_files_without_overwrite(input_images, input_labels, output_images, output_labels)
print("Copying complete.")
