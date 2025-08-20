# detect.py
import torch
import torchvision
import cv2
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from train import get_model_instance_segmentation # Re-use the model definition function

def main():
    # --- Setup ---
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    num_classes = 2 # 1 class (smoke) + background
    MODEL_PATH = "smoke_mask_rcnn_model.pth"
    DATASET_DIR = "./dataset"
    CONFIDENCE_THRESHOLD = 0.3

    # --- Load Model ---
    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval() # Set model to evaluation mode

    # --- Select a random image ---
    image_files = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("No images found in the dataset directory.")
        return
    
    # image_name = random.choice(image_files)
    image_name = "image30.jpg"

    image_path = os.path.join(DATASET_DIR, image_name)
    print(f"Running inference on: {image_path}")

    # --- Run Inference ---
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    img_tensor = transform(img_rgb)

    with torch.no_grad():
        prediction = model([img_tensor.to(device)])

    # --- Visualize the results ---
    scores = prediction[0]['scores'].cpu().numpy()
    boxes = prediction[0]['boxes'].cpu().numpy()
    masks = prediction[0]['masks'].cpu().numpy()

    # Filter predictions by confidence score
    high_confidence_indices = scores > CONFIDENCE_THRESHOLD

    if np.sum(high_confidence_indices) == 0:
        print("No smoke detected with high confidence.")
    else:
        print(f"Detected {np.sum(high_confidence_indices)} instances of smoke.")
        # Create a composite image with masks
        for i, (box, mask) in enumerate(zip(boxes[high_confidence_indices], masks[high_confidence_indices])):
            # Draw mask
            mask_binary = (mask[0] > 0.5)
            color = np.array([0, 0, 255]) # Blue color for mask
            # masked_img = np.where(mask_binary[..., None], color, img)
            # img = cv2.addWeighted(img, 0.5, masked_img, 0.5, 0)
            masked_img = np.where(mask_binary[..., None], color, img).astype(np.uint8)
            img = cv2.addWeighted(img, 0.5, masked_img, 0.5, 0)
            # Draw bounding box
            pt1 = (int(box[0]), int(box[1]))
            pt2 = (int(box[2]), int(box[3]))
            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)

    # Display the final image
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Smoke Detection Result")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()