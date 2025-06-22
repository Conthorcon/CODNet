import torch
from timm.models.pvt_v2 import pvt_v2_b2

# Khởi tạo backbone PVTv2-B2
backbone = pvt_v2_b2(pretrained=True)

# In cấu trúc
print(backbone)

# Tạo input giả
x = torch.randn(1, 3, 224, 224)  # batch_size = 1, RGB, 224x224

# Trích xuất feature
features = backbone.forward_features(x)

# Có 4 scale features, thường ứng dụng trong FPN hoặc decoder
for i, f in enumerate(features):
    print(f"Stage {i+1} output shape: {f.shape}")
