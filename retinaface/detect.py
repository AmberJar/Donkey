import os
import cv2
import time
import argparse
import numpy as np

import torch

from layers import PriorBox
from config import get_config
from models import RetinaFace
from utils.general import draw_detections
from utils.box_utils import decode, decode_landmarks, nms


def parse_arguments():
    parser = argparse.ArgumentParser(description="Inference Arguments for RetinaFace")

    # Model and device options
    parser.add_argument(
        '-w', '--weights',
        type=str,
        default='./weights/resnet50_final.pth',
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
        '--image-path',
        type=str,
        default=r'E:\BaiduNetdiskDownload\donkeys_all_250604\donkeys_all_250604\train\donkey_face',
        help='Path to the input image'
    )
    # Image save
    parser.add_argument(
        '--save-path',
        type=str,
        default=r"E:\BaiduNetdiskDownload\widerface\widerface\result",
        help='Path to the input image'
    )


    return parser.parse_args()


@torch.no_grad()
def inference(model, image):
    model.eval()
    loc, conf, landmarks = model(image)

    loc = loc.squeeze(0)
    conf = conf.squeeze(0)
    landmarks = landmarks.squeeze(0)

    return loc, conf, landmarks




def main(params):
    # load configuration and device setup
    cfg = get_config(params.network)
    if cfg is None:
        raise KeyError(f"Config file for {params.network} not found!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rgb_mean = (124, 135, 133)
    resize_factor = 1

    # model initialization
    model = RetinaFace(cfg=cfg)
    model.to(device)
    model.eval()

    # loading state_dict
    state_dict = torch.load(params.weights, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print("Model loaded successfully!")

    # read image
    for i in os.listdir(params.image_path):
        image_path = os.path.join(params.image_path, i)
        original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if original_image is None:
            print(f"[×] Failed to load image: {image_path}")
            continue

        orig_height, orig_width = original_image.shape[:2]
        resized_image = cv2.resize(original_image, (640, 640))  # align with training input
        image = np.float32(resized_image)
        image -= rgb_mean
        image = image.transpose(2, 0, 1)  # HWC -> CHW
        image = torch.from_numpy(image).unsqueeze(0).to(device)

        # forward pass
        loc, conf, landmarks = inference(model, image)

        # generate anchor boxes
        priorbox = PriorBox(cfg, image_size=(640, 640))
        priors = priorbox.generate_anchors().to(device)

        # decode
        boxes = decode(loc, priors, cfg['variance'])
        landmarks = decode_landmarks(landmarks, priors, cfg['variance'])

        # scale to 640 coords
        boxes = boxes * torch.tensor([640, 640, 640, 640], device=device)
        landmarks = landmarks * torch.tensor([640, 640] * 5, device=device)

        boxes = boxes.cpu().numpy()
        landmarks = landmarks.cpu().numpy()
        scores = conf.cpu().numpy()[:, 1]

        # filter & sort
        inds = scores > params.conf_threshold
        boxes = boxes[inds]
        landmarks = landmarks[inds]
        scores = scores[inds]

        if len(scores) == 0:
            print(f"[×] No detections above threshold in: {i}")
            continue

        order = scores.argsort()[::-1]
        boxes = boxes[order]
        landmarks = landmarks[order]
        scores = scores[order]

        # only keep top-1
        best_box = boxes[0]
        best_landmarks = landmarks[0]
        best_score = scores[0]

        # map back to original image size
        scale_x = orig_width / 640.0
        scale_y = orig_height / 640.0

        best_box[[0, 2]] *= scale_x
        best_box[[1, 3]] *= scale_y
        best_landmarks[0::2] *= scale_x
        best_landmarks[1::2] *= scale_y

        best_box = best_box.astype(int)
        best_landmarks = best_landmarks.reshape(5, 2).astype(int)

        # visualize
        if params.save_image:
            x1, y1, x2, y2 = best_box
            cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for (x, y) in best_landmarks:
                cv2.circle(original_image, (x, y), 20, (0, 0, 255), -1)

            os.makedirs(params.save_path, exist_ok=True)
            name = os.path.splitext(i)[0]
            save_name = os.path.join(params.save_path, f"{name}_top1.jpg")
            cv2.imwrite(save_name, original_image)
            print(f"[✓] Saved: {save_name}")


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
