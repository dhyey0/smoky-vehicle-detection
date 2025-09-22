# preprocess_images.py
import cv2
import os
from glob import glob

SOURCE_DIR = "./new"
DEST_DIR = "./new_resized"
TARGET_WIDTH = 1024 # A good balance of detail and speed

def resize_images():
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
    
    image_paths = glob(os.path.join(SOURCE_DIR, "*.jpg")) + glob(os.path.join(SOURCE_DIR, "*.png"))
    print(f"Found {len(image_paths)} images to resize.")

    for i, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        h, w, _ = img.shape
        target_height = int(h * (TARGET_WIDTH / w))
        resized_img = cv2.resize(img, (TARGET_WIDTH, target_height), interpolation=cv2.INTER_AREA)
        
        filename = os.path.basename(img_path)
        dest_path = os.path.join(DEST_DIR, filename)
        cv2.imwrite(dest_path, resized_img)
        print(f"Processed {i+1}/{len(image_paths)}: {filename}")

    print(f"\nResized images saved in '{DEST_DIR}' folder.")

if __name__ == "__main__":
    resize_images()