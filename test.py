import os
import cv2
import shutil
import numpy as np

# Paths
image_folder = "masks_iou30_bysize_by5componets/passed"
failed_folder = "masks_iou30_bysize_by5componets/failed"
progress_file = "progress.txt"
os.makedirs(failed_folder, exist_ok=True)

# Load processed files from progress file
def load_progress():
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            return f.read().splitlines()
    return []

# Save processed file
def save_progress(image_name):
    with open(progress_file, "a") as f:
        f.write(image_name + "\n")

# Get all image files
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
processed_files = load_progress()

# Start from the first unprocessed image
index = next((i for i, img in enumerate(image_files) if img not in processed_files), len(image_files))
history = []  # Store processed images for undo

while index < len(image_files):
    image_file = image_files[index]
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)
    
    # Get screen resolution
    screen_res = (1920, 1080)  # Adjust if necessary
    black_background = cv2.resize(np.zeros((screen_res[1], screen_res[0], 3), dtype=np.uint8), screen_res)
    
    h, w = image.shape[:2]
    x_offset = (screen_res[0] - w) // 2
    y_offset = (screen_res[1] - h) // 2
    black_background[y_offset:y_offset+h, x_offset:x_offset+w] = image
    
    cv2.namedWindow("Image Review", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Image Review", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Image Review", black_background)

    key = cv2.waitKey(0) & 0xFF

    if key == ord('n'):  # Move to failed folder
        shutil.move(image_path, os.path.join(failed_folder, image_file))
        print(f"Moved {image_file} to failed folder.")
        save_progress(image_file)
        history.append(('n', image_file))
        index += 1
    elif key == ord('y'):  # Keep the image
        print(f"Kept {image_file}.")
        save_progress(image_file)
        history.append(('y', image_file))
        index += 1
    elif key == ord('u') and history:  # Undo last action
        last_action, last_image = history.pop()
        if last_action == 'n':
            shutil.move(os.path.join(failed_folder, last_image), os.path.join(image_folder, last_image))
            print(f"Restored {last_image} from failed folder.")
        index -= 1
    elif key == ord('q'):  # Quit
        print("Quitting program.")
        break

    cv2.destroyAllWindows()

print("Processing complete.")


exit()
import os
import cv2
import shutil

# Paths
image_folder = "masks_iou30_bysize_by5componets/passed"
failed_folder = "masks_iou30_bysize_by5componets/failed"
progress_file = "progress.txt"
os.makedirs(failed_folder, exist_ok=True)

# Load processed files from progress file
def load_progress():
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            return f.read().splitlines()
    return []

# Save processed file
def save_progress(image_name):
    with open(progress_file, "a") as f:
        f.write(image_name + "\n")

# Get all image files
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
processed_files = load_progress()

# Start from the first unprocessed image
index = next((i for i, img in enumerate(image_files) if img not in processed_files), len(image_files))
history = []  # Store processed images for undo

while index < len(image_files):
    image_file = image_files[index]
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)

    cv2.imshow("Image Review", image)

    key = cv2.waitKey(0) & 0xFF

    if key == ord('n'):  # Move to failed folder
        shutil.move(image_path, os.path.join(failed_folder, image_file))
        print(f"Moved {image_file} to failed folder.")
        save_progress(image_file)
        history.append(('n', image_file))
        index += 1
    elif key == ord('y'):  # Keep the image
        print(f"Kept {image_file}.")
        save_progress(image_file)
        history.append(('y', image_file))
        index += 1
    elif key == ord('u') and history:  # Undo last action
        last_action, last_image = history.pop()
        if last_action == 'n':
            shutil.move(os.path.join(failed_folder, last_image), os.path.join(image_folder, last_image))
            print(f"Restored {last_image} from failed folder.")
        index -= 1
    elif key == ord('q'):  # Quit
        print("Quitting program.")
        break

    cv2.destroyAllWindows()

print("Processing complete.")


exit()
import os
import cv2
import shutil

# Paths
image_folder = "masks_iou30_bysize_by5componets/passed"
failed_folder = "masks_iou30_bysize_by5componets/failed"
progress_file = "progress.txt"
os.makedirs(failed_folder, exist_ok=True)

# Zoom & Pan Variables
scale = 1.0
tx, ty = 0, 0
dragging = False
ix, iy = 0, 0
image = None

# Load processed files from progress file
def load_progress():
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            return set(f.read().splitlines())
    return set()

# Save processed file
def save_progress(image_name):
    with open(progress_file, "a") as f:
        f.write(image_name + "\n")

def update_display():
    """ Updates the displayed image with zoom and pan. """
    global image, scale, tx, ty
    
    h, w = image.shape[:2]
    zoomed_w, zoomed_h = int(w * scale), int(h * scale)
    zoomed = cv2.resize(image, (zoomed_w, zoomed_h))

    # Ensure translation does not exceed bounds
    max_tx, max_ty = max(zoomed_w - w, 0), max(zoomed_h - h, 0)
    tx = max(min(tx, max_tx), 0)
    ty = max(min(ty, max_ty), 0)

    # Crop the zoomed image
    cropped = zoomed[ty:ty + h, tx:tx + w]

    cv2.imshow("Image Review", cropped)

def mouse_callback(event, x, y, flags, param):
    """ Handles zooming and panning """
    global scale, tx, ty, dragging, ix, iy

    if event == cv2.EVENT_MOUSEWHEEL:
        # Scroll up to zoom in, down to zoom out
        if flags > 0:  # Scroll up
            scale *= 1.2
        else:  # Scroll down
            scale /= 1.2
        scale = max(min(scale, 5.0), 0.5)  # Limit zoom level
        update_display()

    elif event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        tx += ix - x
        ty += iy - y
        ix, iy = x, y
        update_display()

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False

# Get all image files
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith((".png", ".jpg", ".jpeg"))])
processed_files = load_progress()

# Start from the first unprocessed image
index = next((i for i, img in enumerate(image_files) if img not in processed_files), len(image_files))

while index < len(image_files):
    image_file = image_files[index]
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)

    cv2.namedWindow("Image Review", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Image Review", mouse_callback)

    update_display()

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == ord('n'):  # Move to failed folder
            shutil.move(image_path, os.path.join(failed_folder, image_file))
            print(f"Moved {image_file} to failed folder.")
            save_progress(image_file)
            index += 1
            break
        
        elif key == ord('y'):  # Keep the image
            print(f"Kept {image_file}.")
            save_progress(image_file)
            index += 1
            break
        
        elif key == 81 and index > 0:  # Left Arrow (\u2190) to go back
            print("Going back to previous image.")
            index -= 1
            break
        
        elif key == 83 and index < len(image_files) - 1:  # Right Arrow (\u2192) to go forward
            index += 1
            break
        
        elif key == ord('q'):  # Quit immediately
            print("Exiting program...")
            cv2.destroyAllWindows()
            exit()

    cv2.destroyAllWindows()

print("Processing complete.")


exit()
import os
import cv2
import shutil

# Paths
image_folder = "masks_iou30_bysize_by5componets/passed"
failed_folder = "masks_iou30_bysize_by5componets/failed"
os.makedirs(failed_folder, exist_ok=True)

# Zoom & Pan Variables
scale = 1.0
tx, ty = 0, 0
dragging = False
ix, iy = 0, 0
image = None

def update_display():
    """ Updates the displayed image with zoom and pan. """
    global image, scale, tx, ty
    
    h, w = image.shape[:2]
    zoomed_w, zoomed_h = int(w * scale), int(h * scale)
    zoomed = cv2.resize(image, (zoomed_w, zoomed_h))

    # Ensure translation does not exceed bounds
    max_tx, max_ty = max(zoomed_w - w, 0), max(zoomed_h - h, 0)
    tx = max(min(tx, max_tx), 0)
    ty = max(min(ty, max_ty), 0)

    # Crop the zoomed image
    cropped = zoomed[ty:ty + h, tx:tx + w]

    cv2.imshow("Image Review", cropped)

def mouse_callback(event, x, y, flags, param):
    """ Handles zooming and panning """
    global scale, tx, ty, dragging, ix, iy

    if event == cv2.EVENT_MOUSEWHEEL:
        # Scroll up to zoom in, down to zoom out
        if flags > 0:  # Scroll up
            scale *= 1.2
        else:  # Scroll down
            scale /= 1.2
        scale = max(min(scale, 5.0), 0.5)  # Limit zoom level
        update_display()

    elif event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        tx += ix - x
        ty += iy - y
        ix, iy = x, y
        update_display()

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False

# Get all image files
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)

    cv2.namedWindow("Image Review", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Image Review", mouse_callback)

    update_display()

    key = cv2.waitKey(0) & 0xFF

    if key == ord('n'):
        shutil.move(image_path, os.path.join(failed_folder, image_file))
        print(f"Moved {image_file} to failed folder.")
    elif key == ord('y'):
        print(f"Kept {image_file}.")

    cv2.destroyAllWindows()

print("Processing complete.")


exit()
import os
import cv2
import shutil

# Paths
image_folder = "masks_iou30_bysize_by5componets/passed"
failed_folder = "masks_iou30_bysize_by5componets/failed"

# Ensure the failed folder exists
os.makedirs(failed_folder, exist_ok=True)

# Get all image files
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    
    # Load and display the image
    img = cv2.imread(image_path)
    cv2.imshow("Image Review", img)
    
    # Wait for a key press
    key = cv2.waitKey(0) & 0xFF  # Get the pressed key
    
    if key == ord('n'):  # If 'n' is pressed, move to failed folder
        shutil.move(image_path, os.path.join(failed_folder, image_file))
        print(f"Moved {image_file} to failed folder.")
    
    elif key == ord('y'):  # If 'y' is pressed, keep it
        print(f"Kept {image_file}.")
    
    # Close the image window
    cv2.destroyAllWindows()

print("Processing complete.")


exit()
import cv2
import numpy as np

def check_disjoint_regions(image_path, output_path=None):
    """Finds and highlights black pixel gaps between non-black pixel regions."""
    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Convert grayscale to BGR for visualization
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Get image dimensions
    height, width = image.shape

    # Iterate over rows and columns to find gaps
    for y in range(height):
        row = image[y, :]
        non_black_indices = np.where(row > 0)[0]  # Find non-black pixels in the row
        if len(non_black_indices) > 1:
            for i in range(len(non_black_indices) - 1):
                if np.any(row[non_black_indices[i]:non_black_indices[i + 1]] == 0):
                    color_image[y, non_black_indices[i]:non_black_indices[i + 1]] = [0, 0, 255]  # Mark in Red

    for x in range(width):
        col = image[:, x]
        non_black_indices = np.where(col > 0)[0]  # Find non-black pixels in the column
        if len(non_black_indices) > 1:
            for i in range(len(non_black_indices) - 1):
                if np.any(col[non_black_indices[i]:non_black_indices[i + 1]] == 0):
                    color_image[non_black_indices[i]:non_black_indices[i + 1], x] = [0, 0, 255]  # Mark in Red

    # Display the result
    cv2.imshow("Disjointed Regions", color_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save if output_path is provided
    if output_path:
        cv2.imwrite(output_path, color_image)

# Example usage
image_path = "masks_iou10_bysize_by5componets/failed/EnvironmentA1_frame_00069_jpg.rf.75bfd3fc64139cc1b5e902da38353019_0_0_13.png"  # Replace with actual image path
# output_path = "/path/to/output.png"  # Optional: Save output
check_disjoint_regions(image_path)


exit()
import cv2
import numpy as np

def highlight_black_regions(image_path, output_path=None):
    """Highlights all black (zero-value) pixels in an image."""
    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Create a color version of the image
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Create a mask of black pixels (pixel value = 0)
    black_pixels = (image == 0)

    # Highlight black regions in red
    color_image[black_pixels] = [0, 0, 255]  # Red (BGR format)

    # Display the highlighted image
    cv2.imshow("Highlighted Black Regions", color_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the result if output_path is provided
    if output_path:
        cv2.imwrite(output_path, color_image)

# Example usage
image_path = "masks_iou10_bysize_by5componets\passed\EnvironmentA2_frame_00027_jpg.rf.95953afc96b13f70a3f16149bc59e635_0_0_1.png"  # Replace with actual image path
output_path = "/path/to/output.png"  # Optional: Save output
highlight_black_regions(image_path, output_path)

exit()



import cv2
import numpy as np

def display_contours(image_path):
    """Displays contours of a given image."""
    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Convert to binary mask
    _, binary_mask = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Convert grayscale to BGR for visualization
    contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Draw contours
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    
    # Display the image with contours
    cv2.imshow("Contours", contour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = "masks_iou10_bysize_by5componets\passed\EnvironmentA2_frame_00027_jpg.rf.95953afc96b13f70a3f16149bc59e635_0_0_1.png"  # Replace with actual image path
display_contours(image_path)
