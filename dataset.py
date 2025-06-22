import torch
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
from pycocotools.coco import COCO
import numpy as np

class Dataset(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms=None):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms
        

    def __getitem__(self, idx):
        img, anns = super().__getitem__(idx)
        img_id = self.ids[idx]
        coco = self.coco

        # Convert PIL image to numpy array for Albumentations
        img = np.array(img.convert("RGB"))

        # Bbox format: [x, y, w, h] -> [x1, y1, x2, y2]
        boxes = []
        labels = []


        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])


        # Albumentations expects Pascal VOC format for bboxes (x_min, y_min, x_max, y_max)
        if self._transforms:
            transformed = self._transforms(image=img, bboxes=boxes, category_ids=labels)
            img = transformed['image'].float() / 255.0
            boxes = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
            labels = torch.as_tensor(transformed['category_ids'], dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)


        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
        }

        return img, target