# train.py
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import DataLoader, random_split

from utils import SmokeDataset, get_transform, collate_fn

def get_model_instance_segmentation(num_classes):
    # Load a pre-trained Mask R-CNN model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # Replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

def main():
    # --- Setup ---
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    num_classes = 2  # 1 class (smoke) + background
    DATASET_DIR = "./dataset"
    ANNOTATIONS_FILE = "./annotations/annotations_fixed.json" # Make sure this is pointing to the fixed file

    # --- Dataset and DataLoader ---
    dataset = SmokeDataset(root_dir=DATASET_DIR, annotation_file=ANNOTATIONS_FILE, transforms=get_transform())
    
    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=2)
    # print("a")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    print(f"Training on {len(train_dataset)} images, validating on {len(val_dataset)} images.")

    # --- Model, Optimizer, Scheduler ---
    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    # UPDATED: Lower learning rate
    optimizer = torch.optim.SGD(params, lr=0.0005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # --- Training Loop ---
    num_epochs = 10 # Start with 10 and increase if needed
    print("Starting training...")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for i, (images, targets) in enumerate(train_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            
            # ADDED: Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += losses.item()
            print(f"Epoch {epoch+1}, Iteration {i+1}/{len(train_loader)}, Loss: {losses.item():.4f}")

        lr_scheduler.step()
        print(f"--- Epoch {epoch+1} Summary: Average Loss: {epoch_loss / len(train_loader):.4f} ---")

    # --- Save the Model ---
    torch.save(model.state_dict(), "smoke_mask_rcnn_model.pth")
    print("Training complete. Model saved to smoke_mask_rcnn_model.pth")

if __name__ == "__main__":
    main()