from sklearn import metrics
import numpy as np

def calculate_nme(pred_landmarks, gt_landmarks):
    """Normalized Mean Error (NME) between predicted and ground truth landmarks."""
    assert pred_landmarks.shape == (5, 2)
    assert gt_landmarks.shape == (5, 2)

    eye_dist = np.linalg.norm(gt_landmarks[0] - gt_landmarks[1])
    nme = np.mean(np.linalg.norm(pred_landmarks - gt_landmarks, axis=1)) / (eye_dist + 1e-6)
    return nme

def calculate_iou(box1, box2):
    """Intersection over Union (IoU) between two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

    box_area = (x2 - x1) * (y2 - y1)
    gt_area = (x2g - x1g) * (y2g - y1g)
    union_area = box_area + gt_area - inter_area + 1e-6

    return inter_area / union_area

def calculate_precision_recall(iou_list, iou_threshold=0.5):
    """
    计算 Precision 和 Recall（单目标检测场景）。
    - iou_list: 所有图像的 IoU 值
    - iou_threshold: 判定为 TP 的阈值（通常是 0.5）
s
    每张图一个 GT，一个预测框。
    """
    TP = sum(iou >= iou_threshold for iou in iou_list)
    FP = sum(iou < iou_threshold for iou in iou_list)
    FN = 0  # 假设每张图一定有GT且有预测框，否则需统计漏检

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    return precision, recall

def calculate_auc(nme_list, max_threshold=0.15, step=0.0001):
    """
    AUC of NME curve up to max_threshold.
    nme_list: list of per-image NME values
    max_threshold: usually 0.07 for facial landmark detection
    """
    nme_list = np.array(nme_list)
    nme_list = nme_list[nme_list < 10]  # 过滤掉异常值（同NME计算一致）

    thresholds = np.arange(0.0, max_threshold + step, step)
    accuracies = [(nme_list < t).mean() for t in thresholds]

    auc = metrics.auc(thresholds, accuracies) / max_threshold  # 归一化
    return auc

def evaluate_metrics(preds, gts):
    """
    Evaluate NME, mIoU, and mAP@0.5

    preds: list of dicts with keys 'box' (4,) and 'landmarks' (5,2)
    gts: list of dicts with same format
    """
    nmes, ious = [], []
    for pred, gt in zip(preds, gts):
        nme = calculate_nme(pred["landmarks"], gt["landmarks"])
        iou = calculate_iou(pred["box"], gt["box"])
        nmes.append(nme)
        ious.append(iou)

    mean_nme = np.mean(nmes)
    mean_iou = np.mean(ious)
    mAP_05 = np.mean([iou > 0.5 for iou in ious])

    return {
        "NME": mean_nme,
        "mIoU": mean_iou,
        "mAP@0.5": mAP_05
    }
