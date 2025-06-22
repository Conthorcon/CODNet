from mcod import MCODNet

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator
import json
import os

class MCODDataset:
    def __init__(self, json_file, image_root):
        self.json_file = json_file
        self.image_root = image_root
        self.dataset_dicts = self._load_dataset()
        
    def _load_dataset(self):
        with open(self.json_file, 'r') as f:
            dataset = json.load(f)
            
        dataset_dicts = []
        for img in dataset['images']:
            record = {}
            record['file_name'] = os.path.join(self.image_root, img['file_name'])
            record['image_id'] = img['id']
            record['height'] = img['height']
            record['width'] = img['width']
            
            annos = []
            for ann in dataset['annotations']:
                if ann['image_id'] == img['id']:
                    obj = {
                        'bbox': ann['bbox'],
                        'bbox_mode': BoxMode.XYWH_ABS,
                        'category_id': ann['category_id'],
                        'iscrowd': ann['iscrowd']
                    }
                    annos.append(obj)
            record['annotations'] = annos
            dataset_dicts.append(record)
                
        return dataset_dicts
    
    def __call__(self):
        return self.dataset_dicts

# Đăng ký dataset
def register_mcod_datasets():
    # Đăng ký dataset training
    DatasetCatalog.register(
        "mcod_train",
        MCODDataset(
            "MHCD2022_COCO/train.json",
            "MHCD2022_COCO/train"
        )
    )
    
    # Đăng ký dataset validation
    DatasetCatalog.register(
        "mcod_val",
        MCODDataset(
            "MHCD2022_COCO/val.json",
            "MHCD2022_COCO/val"
        )
    )
    
    # Đăng ký metadata
    for name in ["mcod_train", "mcod_val"]:
        MetadataCatalog.get(name).set(
            thing_classes=["object"],
            evaluator_type="coco"
        )

# Gọi hàm đăng ký
# register_mcod_datasets()

from detectron2.modeling import BACKBONE_REGISTRY, Backbone
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool
from detectron2.layers import ShapeSpec
import timm

import torch.nn as nn
import torch.nn.functional as F

class PVTMCODBackbone(Backbone):
    def __init__(self):
        super().__init__()
        self.pvt = timm.create_model("pvt_v2_b3", pretrained=True, features_only=True)
        channels = [64, 128, 320, 512]
        self.mcod = MCODNet(channels=channels)
        
        # Lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(64, 64, 1),
            nn.Conv2d(128, 64, 1),
            nn.Conv2d(320, 64, 1),
            nn.Conv2d(512, 64, 1)
        ])
        
        # FPN output convolutions
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1)
        ])
        
        # Extra FPN levels
        self.extra_blocks = nn.ModuleList([
            nn.MaxPool2d(kernel_size=1, stride=2, padding=0),
            nn.MaxPool2d(kernel_size=1, stride=2, padding=0)
        ])

    def forward(self, x):
        # Get features from PVT
        feats = self.pvt(x)
        f1, f2, f3, f4 = feats
        
        # Process through MCOD
        mcod_out = self.mcod(f1, f2, f3, f4)
        
        # Build lateral connections
        laterals = [
            self.lateral_convs[0](f1),
            self.lateral_convs[1](f2),
            self.lateral_convs[2](f3),
            self.lateral_convs[3](f4)
        ]
        
        # Build top-down path
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[-2:],
                mode='nearest'
            )
        
        # Build FPN output
        fpn_outputs = []
        for i, lateral in enumerate(laterals):
            fpn_outputs.append(self.fpn_convs[i](lateral))
        
        # Add extra FPN levels
        for extra_block in self.extra_blocks:
            fpn_outputs.append(extra_block(fpn_outputs[-1]))
        
        return {
            "p2": fpn_outputs[0],
            "p3": fpn_outputs[1],
            "p4": fpn_outputs[2],
            "p5": fpn_outputs[3],
            "p6": fpn_outputs[4]
        }

    def output_shape(self):
        return {
            "p2": ShapeSpec(channels=64, stride=4),
            "p3": ShapeSpec(channels=64, stride=8),
            "p4": ShapeSpec(channels=64, stride=16),
            "p5": ShapeSpec(channels=64, stride=32),
            "p6": ShapeSpec(channels=64, stride=64)
        }

@BACKBONE_REGISTRY.register()
def build_pvt_mcod_backbone(cfg, input_shape):
    return PVTMCODBackbone()

def register_mhcd2022_datasets():
    """Register MHCD2022 datasets"""
    # Register training dataset
    DatasetCatalog.register(
        "mhcd2022_train",
        lambda: MCODDataset(
            "MHCD2022_COCO/train.json",
            "MHCD2022_COCO/train"
        )()
    )
    
    # Register validation dataset
    DatasetCatalog.register(
        "mhcd2022_val",
        lambda: MCODDataset(
            "MHCD2022_COCO/val.json",
            "MHCD2022_COCO/val"
        )()
    )
    
    # Set metadata for both datasets
    for name in ["mhcd2022_train", "mhcd2022_val"]:
        MetadataCatalog.get(name).set(
            thing_classes=["person", "aircraft", "military vehicle", "tank"],
            evaluator_type="coco"
        )

def setup_cfg():
    cfg = get_cfg()
    
    # Cấu hình cơ bản
    cfg.DATASETS.TRAIN = ("mhcd2022_train",)
    cfg.DATASETS.TEST = ("mhcd2022_val",)
    
    # Cấu hình model
    cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
    cfg.MODEL.BACKBONE.NAME = "build_pvt_mcod_backbone"
    cfg.MODEL.WEIGHTS = ""  # Start from scratch
    
    # Cấu hình ROI heads
    cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeads"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]
    
    # Cấu hình Box Head
    cfg.MODEL.ROI_BOX_HEAD.NAME = "FastRCNNConvFCHead"
    cfg.MODEL.ROI_BOX_HEAD.NUM_FC = 2
    cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7
    cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 0
    cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignV2"
    cfg.MODEL.ROI_BOX_HEAD.NUM_CONV = 0
    cfg.MODEL.ROI_BOX_HEAD.FC_DIM = 512
    
    # Cấu hình RPN
    cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5"]
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 1000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 500
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 500
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 500
    
    # Cấu hình training
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 10000
    cfg.SOLVER.STEPS = (5000, 7500)
    cfg.SOLVER.GAMMA = 0.1
    
    # Cấu hình evaluation
    cfg.TEST.EVAL_PERIOD = 500
    
    # Cấu hình output
    cfg.OUTPUT_DIR = "output"
    
    # Memory optimization settings
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.MEMORY_EFFICIENT = True
    cfg.MODEL.USE_AMP = True
    
    # Add RPN specific configurations
    cfg.MODEL.RPN.CONV_DIMS = [64]
    cfg.MODEL.RPN.NUM_CONV = 1
    
    return cfg

class MCODTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)

def train_mhcd2022():
    # Đăng ký datasets
    register_mhcd2022_datasets()
    
    # Cấu hình
    cfg = setup_cfg()
    
    # Tạo thư mục output
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Khởi tạo trainer
    trainer = MCODTrainer(cfg)
    
    # Training
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    train_mhcd2022()

# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from detectron2.data import build_detection_test_loader

# evaluator = COCOEvaluator("mcod_val", cfg, False, output_dir="./output_mcod/")
# val_loader = build_detection_test_loader(cfg, "mcod_val")
# inference_on_dataset(trainer.model, val_loader, evaluator)


