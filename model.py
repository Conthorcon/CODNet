import torch
import torch.nn as nn
import torch.nn.functional as F
from Exp.pvtv2 import pvt_v2_b2
import os
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.roi_heads import RoIHeads
from loss import ciou_loss
import math
import torchvision

class PVT(nn.Module):
    def __init__(self):
        super(PVT, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrained_pvt/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

    def forward(self, x):
        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]

        return {"0": x1, "1": x2, "2": x3, "3": x4}
        return x1, x2, x3, x4
class FeatureFusionModule(nn.Module):
    def __init__(self):
        super(FeatureFusionModule, self).__init__()
        c1 = 64
        c2 = 128
        c3 = 320
        c4 = 512

        self.conv3_f4 = nn.Conv2d(c4, c4, kernel_size=3, padding=1)
        self.conv1_f4 = nn.Conv2d(c4, c3, kernel_size=1)
        self.conv3_inter3 = nn.Conv2d(c3, c3, kernel_size=3, padding=1)
        self.conv3_f3 = nn.Conv2d(c3 + c4, c3, kernel_size=3, padding=1)

        self.conv1_f4_f3 = nn.Conv2d(c4 + c3, c2, kernel_size=1)
        self.conv3_inter2 = nn.Conv2d(c2, c2, kernel_size=3, padding=1)
        self.conv3_f2 = nn.Conv2d(c2 + c3, c2, kernel_size=3, padding=1)

        self.conv1_all = nn.Conv2d(c4 + c3 + c2, c1, kernel_size=1)
        self.conv3_inter1 = nn.Conv2d(c1, c1, kernel_size=3, padding=1)
        self.conv3_f1 = nn.Conv2d(c1 + c2, c1, kernel_size=3, padding=1)

        self.conv3_out = nn.Conv2d(c1, c1, kernel_size=3, padding=1)

    def forward(self, f1, f2, f3, f4):
        f4_hat = self.conv3_f4(f4)

        # Upsample layers
        f4_up3 = F.interpolate(f4, size=f3.shape[2:], mode='bilinear', align_corners=False)
        f4_up2 = F.interpolate(f4, size=f2.shape[2:], mode='bilinear', align_corners=False)
        f4_up1 = F.interpolate(f4, size=f1.shape[2:], mode='bilinear', align_corners=False)

        f3_up2 = F.interpolate(f3, size=f2.shape[2:], mode='bilinear', align_corners=False)
        f3_up1 = F.interpolate(f3, size=f1.shape[2:], mode='bilinear', align_corners=False)

        f2_up1 = F.interpolate(f2, size=f1.shape[2:], mode='bilinear', align_corners=False)

        # f3 fusion
        f4_hat_up3 = F.interpolate(f4_hat, size=f3.shape[2:], mode='bilinear', align_corners=False)
        f3_inter = self.conv3_inter3(self.conv1_f4(f4_up3))
        f3_hat = self.conv3_f3(torch.cat([f3_inter * f3 + f3, f4_hat_up3], dim=1))

        # f2 fusion
        f3_hat_up = F.interpolate(f3_hat, size=f2.shape[2:], mode='bilinear', align_corners=False)
        f2_inter = self.conv3_inter2(self.conv1_f4_f3(torch.cat([f4_up2, f3_up2], dim=1)))
        f2_hat = self.conv3_f2(torch.cat([f2_inter * f2 + f2, f3_hat_up], dim=1))

        # f1 fusion
        f2_hat_up = F.interpolate(f2_hat, size=f1.shape[2:], mode='bilinear', align_corners=False)
        f1_inter = self.conv3_inter1(self.conv1_all(torch.cat([f4_up1, f3_up1, f2_up1], dim=1)))
        f1_hat = self.conv3_f1(torch.cat([f1_inter * f1 + f1, f2_hat_up], dim=1))


        f_prime = self.conv3_out(f1_hat)

        return f_prime

class MyRoIHeads(RoIHeads):
    def fastrcnn_loss(self, class_logits, box_regression, labels, regression_targets):
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        # --- Classification Loss: BCE ---
        target_onehot = F.one_hot(labels, num_classes=class_logits.shape[1]).float()
        loss_classifier = F.binary_cross_entropy_with_logits(class_logits, target_onehot)

        # --- Regression Loss: CIoU ---
        pos_inds = torch.where(labels > 0)[0]
        pred_boxes = box_regression[pos_inds]
        target_boxes = regression_targets[pos_inds]
        loss_box_reg = ciou_loss(pred_boxes, target_boxes).mean()

        return loss_classifier, loss_box_reg

class MyFasterRCNN(FasterRCNN):
    def __init__(self,num_classes=5):

        resnet = models.resnet50(pretrained=True)
        
        # Bỏ phần cuối (avgpool và fc)
        backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Đặt thuộc tính out_channels để FasterRCNN biết số kênh output
        backbone.out_channels = 2048

        # Tạo anchor generator cho tầng feature duy nhất
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )

        roi_pooler = MultiScaleRoIAlign(
            featmap_names=["0"], 
            output_size=7, 
            sampling_ratio=2
        )


        super().__init__(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
        )

        self.roi_heads = MyRoIHeads(
            box_roi_pool=self.roi_heads.box_roi_pool,
            box_head=self.roi_heads.box_head,
            box_predictor=self.roi_heads.box_predictor,
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.25,
            bbox_reg_weights=None,
            score_thresh=0.05,             # Threshold để giữ lại các box có score đủ cao
            nms_thresh=0.5,                # Threshold cho Non-Max Suppression
            detections_per_img=100        # Số box tối đa mỗi ảnh sau NMS
        )






