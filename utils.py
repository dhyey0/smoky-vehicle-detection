# utils.py
import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import torchvision.transforms as T

class SmokeDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Get image info
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotations = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        
        # Open image
        img_path = os.path.join(self.root_dir, path.split('/')[-1])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Number of objects
        num_objs = len(coco_annotations)
        
        # Bounding boxes for objects
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotations[i]['bbox'][0]
            ymin = coco_annotations[i]['bbox'][1]
            xmax = xmin + coco_annotations[i]['bbox'][2]
            ymax = ymin + coco_annotations[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # Labels (there is only one class: smoke)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        
        # Masks
        masks = []
        for ann in coco_annotations:
            mask = coco.annToMask(ann)
            masks.append(mask)
        masks = np.array(masks)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # Create the target dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = torch.tensor([img_id])
        
        # --- CHANGED SECTION ---
        # Apply transforms only to the image
        if self.transforms is not None:
            img = self.transforms(img)
            
        return img, target

    def __len__(self):
        return len(self.ids)

# --- UPDATED get_transform function ---
def get_transform():
    # This now returns a standard transform pipeline that only processes the image
    return T.Compose([
        T.ToTensor()
    ])

# This is needed for collate_fn to handle lists of targets
def collate_fn(batch):
    return tuple(zip(*batch))