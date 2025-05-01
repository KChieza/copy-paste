import os
import cv2
import shutil
import numpy as np
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Image review tool for sorting images into passed/failed folders.')
    parser.add_argument('--input', required=True, help='Path to the folder containing images to review')
    parser.add_argument('--failed', required=True, help='Path to the folder where failed images will be moved')
    parser.add_argument('--progress', default='progress.txt', help='Path to the progress tracking file')
    parser.add_argument('--screen_width', type=int, default=1920, help='Screen width for display')
    parser.add_argument('--screen_height', type=int, default=1080, help='Screen height for display')
    return parser.parse_args()

# Load processed files from progress file
def load_progress(progress_file):
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            return f.read().splitlines()
    return []

# Save processed file
def save_progress(progress_file, image_name):
    with open(progress_file, "a") as f:
        f.write(image_name + "\n")

def main():
    args = parse_arguments()
    
    # Create failed folder if it doesn't exist
    os.makedirs(args.failed, exist_ok=True)

    # Load processed files from progress file
    processed_files = load_progress(args.progress)

    # Get all image files
    image_files = sorted([f for f in os.listdir(args.input) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    # Start from the first unprocessed image
    index = next((i for i, img in enumerate(image_files) if img not in processed_files), len(image_files))
    history = []  # Store processed images for undo

    while index < len(image_files):
        image_file = image_files[index]
        image_path = os.path.join(args.input, image_file)
        image = cv2.imread(image_path)
        
        # Create black background with specified screen resolution
        screen_res = (args.screen_width, args.screen_height)
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
            shutil.move(image_path, os.path.join(args.failed, image_file))
            print(f"Moved {image_file} to failed folder.")
            save_progress(args.progress, image_file)
            history.append(('n', image_file))
            index += 1
        elif key == ord('y'):  # Keep the image
            print(f"Kept {image_file}.")
            save_progress(args.progress, image_file)
            history.append(('y', image_file))
            index += 1
        elif key == ord('u') and history:  # Undo last action
            last_action, last_image = history.pop()
            if last_action == 'n':
                shutil.move(os.path.join(args.failed, last_image), os.path.join(args.input, last_image))
                print(f"Restored {last_image} from failed folder.")
            index -= 1
        elif key == ord('q'):  # Quit
            print("Quitting program.")
            break

        cv2.destroyAllWindows()

    print("Processing complete.")

if __name__ == "__main__":
    main()