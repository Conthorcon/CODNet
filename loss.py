import torch
import torch.nn.functional as F

def ciou_loss(pred_boxes, target_boxes):
    # pred_boxes, target_boxes: [N, 4] format [x1, y1, x2, y2]
    # CIoU Loss Implementation (simplified version)
    import math

    pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes.unbind(1)
    target_x1, target_y1, target_x2, target_y2 = target_boxes.unbind(1)

    pred_w = pred_x2 - pred_x1
    pred_h = pred_y2 - pred_y1
    pred_cx = (pred_x1 + pred_x2) / 2
    pred_cy = (pred_y1 + pred_y2) / 2

    target_w = target_x2 - target_x1
    target_h = target_y2 - target_y1
    target_cx = (target_x1 + target_x2) / 2
    target_cy = (target_y1 + target_y2) / 2

    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)

    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union_area = pred_w * pred_h + target_w * target_h - inter_area
    iou = inter_area / union_area.clamp(min=1e-6)

    center_dist = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2
    enclose_x1 = torch.min(pred_x1, target_x1)
    enclose_y1 = torch.min(pred_y1, target_y1)
    enclose_x2 = torch.max(pred_x2, target_x2)
    enclose_y2 = torch.max(pred_y2, target_y2)
    c2 = ((enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2).clamp(min=1e-6)

    v = (4 / (math.pi ** 2)) * (torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h)) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-6)

    ciou = iou - (center_dist / c2 + alpha * v)
    return 1.0 - ciou  # CIoU loss
