import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Feature Fusion Module (FFM) ---
class FeatureFusionModule(nn.Module):
    def __init__(self, channels):
        super(FeatureFusionModule, self).__init__()
        c1, c2, c3, c4 = channels

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
        f_hat4 = self.conv3_f4(f4)

        f4_up = F.interpolate(f4, size=f3.shape[2:], mode='bilinear', align_corners=False)
        f_inter3 = self.conv3_inter3(self.conv1_f4(f4_up))
        f_hat4_up = F.interpolate(f_hat4, size=f3.shape[2:], mode='bilinear', align_corners=False)
        f_hat3 = self.conv3_f3(torch.cat([f_inter3 * f3 + f3, f_hat4_up], dim=1))

        f4_up2 = F.interpolate(f4, size=f2.shape[2:], mode='bilinear', align_corners=False)
        f3_up2 = F.interpolate(f3, size=f2.shape[2:], mode='bilinear', align_corners=False)
        f_inter2 = self.conv3_inter2(self.conv1_f4_f3(torch.cat([f4_up2, f3_up2], dim=1)))
        f_hat3_up = F.interpolate(f_hat3, size=f2.shape[2:], mode='bilinear', align_corners=False)
        f_hat2 = self.conv3_f2(torch.cat([f_inter2 * f2 + f2, f_hat3_up], dim=1))

        f4_up1 = F.interpolate(f4, size=f1.shape[2:], mode='bilinear', align_corners=False)
        f3_up1 = F.interpolate(f3, size=f1.shape[2:], mode='bilinear', align_corners=False)
        f2_up1 = F.interpolate(f2, size=f1.shape[2:], mode='bilinear', align_corners=False)
        f_inter1 = self.conv3_inter1(self.conv1_all(torch.cat([f4_up1, f3_up1, f2_up1], dim=1)))
        f_hat2_up = F.interpolate(f_hat2, size=f1.shape[2:], mode='bilinear', align_corners=False)
        f_hat1 = self.conv3_f1(torch.cat([f_inter1 * f1 + f1, f_hat2_up], dim=1))

        out = self.conv3_out(f_hat1)
        return out

# --- Boundary Extraction Module (BEM) ---
class BoundaryExtractionModule(nn.Module):
    def __init__(self, f1_channels, fused_channels):
        super(BoundaryExtractionModule, self).__init__()
        self.conv1_f1 = nn.Conv2d(f1_channels, fused_channels, kernel_size=1)
        self.conv1_fused = nn.Conv2d(fused_channels, fused_channels, kernel_size=1)
        self.conv3_1 = nn.Conv2d(fused_channels * 2, fused_channels, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(fused_channels, fused_channels, kernel_size=3, padding=1)
        self.conv1_out = nn.Conv2d(fused_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, f1, f_fused):
        f_fused_resized = F.interpolate(f_fused, size=f1.shape[2:], mode='bilinear', align_corners=False)
        f1_ = self.conv1_f1(f1)
        f_fused_ = self.conv1_fused(f_fused_resized)
        x = torch.cat([f1_, f_fused_], dim=1)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv1_out(x)
        fb = self.sigmoid(x)
        return fb

# --- Boundary Guidance Module (BGM) ---
class BoundaryGuidanceModule(nn.Module):
    def __init__(self, in_channels):
        super(BoundaryGuidanceModule, self).__init__()
        self.conv3_fused = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        k = int(abs((math.log2(in_channels) + 1) / 2))
        k = k if k % 2 == 1 else k + 1
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2)
        self.conv1_reduce = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fi, fb):
        fb_down = F.interpolate(fb, size=fi.shape[2:], mode='bilinear', align_corners=False)
        fused = (fi * fb_down) + fi
        fused = self.conv3_fused(fused)
        gap = self.global_pool(fused)
        attn = self.conv1d(gap.view(gap.size(0), 1, -1))
        attn = self.sigmoid(attn).view(gap.size(0), -1, 1, 1)
        out = self.conv1_reduce(fused * attn)
        return out

# --- Context Fusion Module (CFM) ---
class ContextFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ContextFusionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1)
        self.branches = nn.ModuleList([
            nn.Conv2d(in_channels // 4, in_channels // 4, 3, padding=d, dilation=d)
            for d in (1, 2, 3, 4)
        ])
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, fa_low, fa_high):
        fa_high_up = F.interpolate(fa_high, size=fa_low.shape[2:], mode='bilinear', align_corners=False)
        fused = torch.cat([fa_low, fa_high_up], dim=1)
        fused = self.conv1(fused)
        chunks = torch.chunk(fused, 4, dim=1)
        out_chunks = []
        for i, branch in enumerate(self.branches):
            left = chunks[i - 1] if i > 0 else chunks[i]
            mid = chunks[i]
            right = chunks[i + 1] if i < 3 else chunks[i]
            out_chunks.append(branch(left + mid + right))
        out = torch.cat(out_chunks, dim=1)
        return self.conv3(out + fused)

# --- Toàn bộ kiến trúc ---
class MCODNet(nn.Module):
    def __init__(self, channels=(64, 128, 320, 512)):
        super(MCODNet, self).__init__()
        c1, c2, c3, c4 = channels

        self.ffm = FeatureFusionModule(channels)
        self.bem = BoundaryExtractionModule(c1, c1)

        self.bgm1 = BoundaryGuidanceModule(c1)
        self.bgm2 = BoundaryGuidanceModule(c2)
        self.bgm3 = BoundaryGuidanceModule(c3)
        self.bgm4 = BoundaryGuidanceModule(c4)

        self.cfm3 = ContextFusionModule(c3 + c4, c3)
        self.cfm2 = ContextFusionModule(c2 + c3, c2)
        self.cfm1 = ContextFusionModule(c1 + c2, c1)

        self.final_pred = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, f1, f2, f3, f4):
        f_fused = self.ffm(f1, f2, f3, f4)
        fb = self.bem(f1, f_fused)

        fa1 = self.bgm1(f1, fb)
        fa2 = self.bgm2(f2, fb)
        fa3 = self.bgm3(f3, fb)
        fa4 = self.bgm4(f4, fb)

        fc3 = self.cfm3(fa3, fa4)
        fc2 = self.cfm2(fa2, fc3)
        fc1 = self.cfm1(fa1, fc2)

        out = self.final_pred(fc1)
        return out

import torch
import torch.nn as nn
import timm

class MCODModel(nn.Module):
    def __init__(self, pretrained=True):
        super(MCODModel, self).__init__()

        # Load backbone PVT-V2-B3 từ timm
        self.backbone = timm.create_model('pvt_v2_b3', pretrained=pretrained, features_only=True)

        # Get channel dimensions from backbone
        channels = [feat['num_chs'] for feat in self.backbone.feature_info]  # [64, 128, 320, 512]

        # Gắn vào MCODNet
        self.mcod_head = MCODNet(channels=tuple(channels))

    def forward(self, x):
        # Trích xuất đặc trưng từ backbone
        features = self.backbone(x)  # list: [f1, f2, f3, f4]
        f1, f2, f3, f4 = features
        out = self.mcod_head(f1, f2, f3, f4)
        return out


def binary_cross_entropy_loss(pred, target):
    return F.binary_cross_entropy_with_logits(pred, target)

def bbox_ciou_loss(pred_box, target_box):
    """
    pred_box, target_box: [B, 4] format: [cx, cy, w, h]
    """
    pred_x, pred_y, pred_w, pred_h = pred_box[:, 0], pred_box[:, 1], pred_box[:, 2], pred_box[:, 3]
    target_x, target_y, target_w, target_h = target_box[:, 0], target_box[:, 1], target_box[:, 2], target_box[:, 3]

    # IoU
    pred_x1 = pred_x - pred_w / 2
    pred_y1 = pred_y - pred_h / 2
    pred_x2 = pred_x + pred_w / 2
    pred_y2 = pred_y + pred_h / 2

    target_x1 = target_x - target_w / 2
    target_y1 = target_y - target_h / 2
    target_x2 = target_x + target_w / 2
    target_y2 = target_y + target_h / 2

    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union_area = (pred_w * pred_h + target_w * target_h) - inter_area
    iou = inter_area / (union_area + 1e-6)

    # Center distance
    center_dist = (pred_x - target_x) ** 2 + (pred_y - target_y) ** 2

    # Enclosing box diagonal
    enclose_x1 = torch.min(pred_x1, target_x1)
    enclose_y1 = torch.min(pred_y1, target_y1)
    enclose_x2 = torch.max(pred_x2, target_x2)
    enclose_y2 = torch.max(pred_y2, target_y2)
    diag = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2

    # Aspect ratio penalty
    v = (4 / (torch.pi ** 2)) * torch.pow(torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-6)
    ciou = iou - center_dist / (diag + 1e-6) - alpha * v
    return 1 - ciou.mean()

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        images = batch['image'].to(device)           # (B, 3, H, W)
        masks = batch['mask'].to(device)             # (B, 1, H, W)

        optimizer.zero_grad()
        outputs = model(images)                      # logits (B, 1, H, W)

        # Resize mask nếu cần
        if outputs.shape != masks.shape:
            masks = F.interpolate(masks, size=outputs.shape[2:], mode='bilinear', align_corners=False)

        loss = binary_cross_entropy_loss(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

from torch.utils.data import DataLoader

# Giả sử bạn có Dataset trả về {'image': tensor, 'mask': tensor}
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

model = MCODModel().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0025, momentum=0.9, weight_decay=0.0001)

for epoch in range(1, 13):
    loss = train_one_epoch(model, train_loader, optimizer, device)
    print(f"Epoch {epoch}, Loss: {loss:.4f}\")