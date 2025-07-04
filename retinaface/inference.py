import os
import cv2
import time
import argparse
import numpy as np
from tqdm import tqdm
import torch
from utils.dataset import WiderFaceDetection
from layers import PriorBox
from config import get_config
from models import RetinaFace
from utils.general import draw_detections
from utils.box_utils import decode, decode_landmarks, nms
from torch.utils.data import DataLoader
from utils.transform import Augmentation
from utils.metrics import evaluate_metrics,calculate_nme,calculate_iou,calculate_auc,calculate_precision_recall

def parse_arguments():
    parser = argparse.ArgumentParser(description="Inference Arguments for RetinaFace")

    # Model and device options
    parser.add_argument(
        '-w', '--weights',
        type=str,
        default='./weights/mobilenetv2_final.pth',
        help='Path to the trained model weights'
    )
    parser.add_argument(
        '-n', '--network',
        type=str,
        default='mobilenetv2',
        choices=[
            'mobilenetv1', 'mobilenetv1_0.25', 'mobilenetv1_0.50',
            'mobilenetv2', 'resnet50', 'resnet34', 'resnet18'
        ],
        help='Backbone network architecture to use'
    )

    # Detection settings
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.02,
        help='Confidence threshold for filtering detections'
    )
    parser.add_argument(
        '--pre-nms-topk',
        type=int,
        default=5000,
        help='Maximum number of detections to consider before applying NMS'
    )
    parser.add_argument(
        '--nms-threshold',
        type=float,
        default=0.4,
        help='Non-Maximum Suppression (NMS) threshold'
    )
    parser.add_argument(
        '--post-nms-topk',
        type=int,
        default=750,
        help='Number of highest scoring detections to keep after NMS'
    )

    # Output options
    parser.add_argument(
        '-s', '--save-image',
        action='store_true',
        help='Save the detection results as images'
    )
    parser.add_argument(
        '-v', '--vis-threshold',
        type=float,
        default=0.6,
        help='Visualization threshold for displaying detections'
    )

    # Image input
    parser.add_argument(
        '--val-data',
        type=str,
        default=r'E:\BaiduNetdiskDownload\donkeys_all_250604\donkeys_all_250604\train',
        help='Path to the input image'
    )
    # Image save
    parser.add_argument(
        '--save-path',
        type=str,
        default=r"E:\BaiduNetdiskDownload\widerface\widerface\result",
        help='Path to the input image'
    )
    parser.add_argument('--num-workers', default=0, type=int, help='Number of workers to use for data loading.')
    parser.add_argument('--batch-size', default=1, type=int, help='Number of samples in each batch during training.')

    return parser.parse_args()


@torch.no_grad()
def inference(model, image):
    model.eval()
    loc, conf, landmarks = model(image)
    return loc.squeeze(0), conf.squeeze(0), landmarks.squeeze(0)


def main(params):
    cfg = get_config(params.network)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rgb_mean = (124, 135, 133)

    # 模型初始化
    model = RetinaFace(cfg=cfg).to(device).eval()
    state_dict = torch.load(params.weights, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print("Model loaded successfully!")

    # 数据加载
    dataset = WiderFaceDetection(params.val_data, Augmentation(cfg['image_size'], rgb_mean))
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False,
                             num_workers=params.num_workers, collate_fn=dataset.collate_fn,
                             pin_memory=True, drop_last=False)
    print("Data loaded successfully!")

    priorbox = PriorBox(cfg, image_size=(640, 640))
    priors = priorbox.generate_anchors().to(device)


    nme_list = []
    iou_list = []
    ap_list = []
    for batch_idx, (images, targets) in enumerate(tqdm(data_loader)):
        images = images.to(device)
        target = targets[0].cpu().numpy()  # 只处理batch中的第一张

        if target.shape[0] == 0:
            continue

        # 推理
        loc, conf, landmarks = inference(model, images)
        boxes = decode(loc, priors, cfg['variance'])
        landms = decode_landmarks(landmarks, priors, cfg['variance'])

        # 还原回640大小
        boxes *= torch.tensor([640, 640, 640, 640], device=device)
        landms *= torch.tensor([640, 640] * 5, device=device)

        boxes = boxes.cpu().numpy()
        landms = landms.cpu().numpy()
        scores = conf.cpu().numpy()[:, 1]

        # 筛选分数
        inds = scores > params.conf_threshold
        if not np.any(inds):
            continue
        boxes, landms, scores = boxes[inds], landms[inds], scores[inds]

        # 取最高分预测框
        order = scores.argsort()[::-1]
        box = boxes[order[0]]
        landmark = landms[order[0]].reshape(5, 2)

        gt_box = target[0, 0:4] * 640  # 归一化坐标 * 尺寸
        gt_landmark = target[0, 4:14].reshape(5, 2) * 640


        nme = calculate_nme(landmark, gt_landmark)
        iou = calculate_iou(box, gt_box)
        ap = 1 if iou > 0.5 else 0

        print(f"[✓] Image {batch_idx:03d}: NME={nme:.4f}, IoU={iou:.4f}, AP@0.5={ap}")

        # 汇总
        if nme < 10:  # 正常范围一般 <1，保守设为10过滤爆炸值
            nme_list.append(nme)
        else:
            print(f"[!] Skipped abnormal NME in image {batch_idx:03d}: {nme:.4f}")
        iou_list.append(iou)
        ap_list.append(ap)

    # 最终结果
    print("\n===== Final Evaluation =====")
    print(f"Mean NME     : {np.mean(nme_list):.4f}")
    precision, recall = calculate_precision_recall(iou_list, iou_threshold=0.5)
    print(f"Precision@0.5: {precision:.4f}")
    print(f"Recall@0.5   : {recall:.4f}")
    print(f"Mean IoU     : {np.mean(iou_list):.4f}")
    print(f"mAP@0.5      : {np.mean(ap_list):.4f}")
    print(f"AUC@0.08     : {calculate_auc(nme_list):.4f}")

        # # 可视化
        # if params.save_image:
        #     orig_img = images[0].permute(1, 2, 0).cpu().numpy() + np.array(rgb_mean)
        #     orig_img = np.clip(orig_img, 0, 255).astype(np.uint8).copy()
        #
        #     scale_x = orig_img.shape[1] / 640.0
        #     scale_y = orig_img.shape[0] / 640.0
        #
        #     vis_box = box.copy()
        #     vis_landmark = landmark.copy()
        #     vis_box[[0, 2]] *= scale_x
        #     vis_box[[1, 3]] *= scale_y
        #     vis_landmark[0::2] *= scale_x
        #     vis_landmark[1::2] *= scale_y
        #
        #     vis_box = vis_box.astype(int)
        #     vis_landmark = vis_landmark.reshape(5, 2).astype(int)
        #
        #     cv2.rectangle(orig_img, (vis_box[0], vis_box[1]), (vis_box[2], vis_box[3]), (0, 255, 0), 2)
        #     for (x, y) in vis_landmark:
        #         cv2.circle(orig_img, (x, y), 3, (0, 0, 255), -1)
        #
        #     os.makedirs(params.save_path, exist_ok=True)
        #     save_path = os.path.join(params.save_path, f"vis_{batch_idx}.jpg")
        #     cv2.imwrite(save_path, orig_img)



if __name__ == '__main__':
    args = parse_arguments()
    main(args)
