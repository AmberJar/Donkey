import os
import cv2
import argparse
import numpy as np
import torch

from retinaface.layers import PriorBox
from retinaface.config import get_config
from retinaface.models import RetinaFace
from retinaface.utils.box_utils import decode, decode_landmarks, nms


def parse_arguments():
    parser = argparse.ArgumentParser(description="Inference Arguments for RetinaFace")
    parser.add_argument('-w', '--weights', type=str, default='./weights/resnet50_final.pth')
    parser.add_argument('-n', '--network', type=str, default='mobilenetv2')
    parser.add_argument('--conf-threshold', type=float, default=0.02)
    parser.add_argument('--nms-threshold', type=float, default=0.4)
    parser.add_argument('--image-path', type=str, default=r'./images')
    parser.add_argument('--save-path', type=str, default=r'./results')
    parser.add_argument('--save-image', action='store_true')
    parser.add_argument('--vis-threshold', type=float, default=0.6)
    return parser.parse_args()


@torch.no_grad()
def inference(model, image):
    model.eval()
    loc, conf, landmarks = model(image)
    return loc.squeeze(0), conf.squeeze(0), landmarks.squeeze(0)


def get_landmarks(params):
    cfg = get_config(params.network)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rgb_mean = (124, 135, 133)

    model = RetinaFace(cfg=cfg)
    model.load_state_dict(torch.load(params.weights, map_location=device))
    model = model.to(device).eval()

    os.makedirs(params.save_path, exist_ok=True)

    for fname in os.listdir(params.image_path):
        image_path = os.path.join(params.image_path, fname)
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"[×] Cannot read image: {image_path}")
            continue

        orig_h, orig_w = original_image.shape[:2]
        resized_image = cv2.resize(original_image, (640, 640))
        image = resized_image.astype(np.float32) - rgb_mean
        image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(device)

        loc, conf, landmarks = inference(model, image)
        priorbox = PriorBox(cfg, image_size=(640, 640))
        priors = priorbox.generate_anchors().to(device)

        boxes = decode(loc, priors, cfg['variance'])
        landmarks = decode_landmarks(landmarks, priors, cfg['variance'])

        boxes *= torch.tensor([640, 640, 640, 640], device=device)
        landmarks *= torch.tensor([640, 640] * 5, device=device)

        scores = conf[:, 1]
        valid_mask = scores > params.conf_threshold
        boxes = boxes[valid_mask]
        landmarks = landmarks[valid_mask]
        scores = scores[valid_mask]

        if len(scores) == 0:
            print(f"[×] No faces detected in: {fname}")
            continue

        keep = nms(boxes, scores, params.nms_threshold)
        boxes = boxes[keep]
        landmarks = landmarks[keep]
        scores = scores[keep]

        # sort by score and take top-1
        order = scores.argsort(descending=True)
        best_box = boxes[order[0]].cpu().numpy()
        best_landmarks = landmarks[order[0]].cpu().numpy()

        # rescale to original image
        scale_x, scale_y = orig_w / 640.0, orig_h / 640.0
        best_box[[0, 2]] *= scale_x
        best_box[[1, 3]] *= scale_y
        best_landmarks[0::2] *= scale_x
        best_landmarks[1::2] *= scale_y

        best_box = best_box.astype(int)
        best_landmarks = best_landmarks.reshape(5, 2).astype(int)

        # Optional save visualization
        if params.save_image:
            vis_image = original_image.copy()
            x1, y1, x2, y2 = best_box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for (x, y) in best_landmarks:
                cv2.circle(vis_image, (x, y), 3, (0, 0, 255), -1)
            out_path = os.path.join(params.save_path, fname)
            cv2.imwrite(out_path, vis_image)
            print(f"[✓] Saved result: {out_path}")

        return best_box, best_landmarks  # return top-1


def main():
    args = parse_arguments()
    best_box, best_landmarks = get_landmarks(args)
    print(f"Best Box: {best_box}")
    print(f"Best Landmarks: {best_landmarks}")


if __name__ == '__main__':
    main()
