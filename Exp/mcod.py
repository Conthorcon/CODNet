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

