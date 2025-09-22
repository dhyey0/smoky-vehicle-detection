#check_dataset.py
import os
import json
import cv2

ANNOTATIONS_FILE = "./annotations/annotations.json"
DATASET_DIR = "./dataset_resized"

def check_dataset_integrity():
    """
    Checks if all images listed in the COCO annotations file
    exist and are readable in the dataset directory.
    """
    found_issues = False
    print("--- Starting Dataset Integrity Check ---")

    try:
        with open(ANNOTATIONS_FILE, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Annotation file not found at: {ANNOTATIONS_FILE}")
        return

    images_info = data.get('images', [])
    if not images_info:
        print("WARNING: No images found in the annotation file.")
        return

    for image_info in images_info:
        filename = image_info.get('file_name', '').split('/')[-1]
        image_path = os.path.join(DATASET_DIR, filename)

        # 1. Check if the file exists
        if not os.path.exists(image_path):
            print(f"ISSUE FOUND: File listed in annotations does not exist: {image_path}")
            found_issues = True
            continue # No need to try reading it if it doesn't exist

        # 2. Check if the file is readable by OpenCV
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"ISSUE FOUND: File exists but is corrupted or unreadable: {image_path}")
                found_issues = True
        except Exception as e:
            print(f"ISSUE FOUND: An error occurred while reading file {image_path}: {e}")
            found_issues = True

    if not found_issues:
        print("\n✅ --- Check Complete: No issues found. Your dataset looks good! ---")
    else:
        print("\n❌ --- Check Complete: Issues were found. Please fix the files listed above. ---")


if __name__ == "__main__":
    check_dataset_integrity()