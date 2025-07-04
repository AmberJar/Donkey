import os
import sys

# 获取当前文件的父目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))  # embedding 的上一级

# 添加到 sys.path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms as T
from skimage import transform as trans
from insightface.recognition.arcface_torch.backbones import get_model
from tqdm import tqdm


# def parse_retinaface_line(line):
#     parts = line.strip().split()
#     if len(parts) != 21:
#         return None, None, None

#     filename = parts[0]
#     x1, y1 = float(parts[1]), float(parts[2])
#     w, h = float(parts[3]), float(parts[4])
#     x2, y2 = x1 + w, y1 + h

#     bbox = [int(max(0, x1)), int(max(0, y1)), int(x2), int(y2)]

#     landmarks = np.array([
#         [float(parts[5]), float(parts[6])],
#         [float(parts[8]), float(parts[9])],
#         [float(parts[11]), float(parts[12])],
#         [float(parts[14]), float(parts[15])],
#         [float(parts[17]), float(parts[18])],
#     ], dtype=np.float32)

#     return filename, bbox, landmarks


def parse_retinaface_line(line):
    parts = line.strip().split()

    # 动态确定 filename 部分有多少 token
    for i in range(1, len(parts)):
        maybe_numbers = parts[i:]
        if len(maybe_numbers) == 20 and all(p.replace('.', '', 1).replace('-', '', 1).isdigit() for p in maybe_numbers):
            filename = ' '.join(parts[:i])
            num_values = maybe_numbers
            break
    else:
        raise ValueError(f"Line format error: cannot locate 20 float fields in: {line}")

    assert len(num_values) == 20, f"Expected 20 float values, got {len(num_values)} → {line}"

    x1, y1 = float(num_values[0]), float(num_values[1])
    w, h = float(num_values[2]), float(num_values[3])
    x2, y2 = x1 + w, y1 + h
    bbox = [int(max(0, x1)), int(max(0, y1)), int(x2), int(y2)]

    landmarks = np.array([
        [float(num_values[4]), float(num_values[5])],
        [float(num_values[7]), float(num_values[8])],
        [float(num_values[10]), float(num_values[11])],
        [float(num_values[13]), float(num_values[14])],
        [float(num_values[16]), float(num_values[17])],
    ], dtype=np.float32)

    return filename, bbox, landmarks

def get_dynamic_template(landmarks, output_size=(112, 112), scale=1.2):
    center = np.mean(landmarks, axis=0)
    landmarks_centered = landmarks - center
    max_dist = np.max(np.linalg.norm(landmarks_centered, axis=1))
    scaled = (landmarks_centered / max_dist) * (output_size[0] / 2 / scale) + output_size[0] / 2
    return scaled


def align_face(img, bbox, landmarks, image_size=(112, 112)):
    x1, y1, x2, y2 = bbox
    face = img[y1:y2, x1:x2]
    if face.shape[0] == 0 or face.shape[1] == 0:
        return None

    landmarks_local = landmarks - np.array([x1, y1], dtype=np.float32)
    dst = get_dynamic_template(landmarks_local, output_size=image_size)
    tform = trans.SimilarityTransform()
    tform.estimate(landmarks_local, dst)
    M = tform.params[0:2, :]
    aligned = cv2.warpAffine(face, M, image_size, borderValue=0.0)

    return aligned


def extract_label(filename):
    return "_".join(Path(filename).stem.split("_")[:-1])


def build_feature_library(images_dir, txt_file, save_path, model_ckpt, device="cuda"):
    model = get_model('vit_b', fp16=False).to(device)
    model.load_state_dict(torch.load(model_ckpt))
    model.eval()

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    features = []
    labels = []

    with open(txt_file, 'r') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Building feature library"):
        filename, bbox, landmarks = parse_retinaface_line(line)
        if filename is None:
            print(f"{filename} not found!!")
            continue

        img_path = Path(images_dir) / filename
        if not img_path.exists():
            print(f"{img_path} not exist!!")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"{img} is None!!")
            continue

        aligned = align_face(img, bbox, landmarks)
        if aligned is None:
            print(f"aligned is None!!")
            continue

        image = Image.fromarray(aligned)
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = model(image_tensor)
            feat = F.normalize(feat, p=2, dim=1)

        features.append(feat.cpu().numpy())
        labels.append(extract_label(filename))

    features = np.vstack(features)
    print(f"[INFO] Saving {len(labels)} embeddings to {save_path}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, features=features, labels=labels)


if __name__ == "__main__":
    images_dir = "/scratch/pf2m24/data/Donkey_xiabao_face/images"
    txt_file = "/scratch/pf2m24/data/Donkey_xiabao_face/val_annotations.txt"
    save_path = "./output_features/features_and_labels_face_val_vitb.npz"
    model_ckpt = "/scratch/pf2m24/projects/donkey_place/insightface/recognition/arcface_torch/work_dirs/donkey_vit_b/model.pt"

    build_feature_library(images_dir, txt_file, save_path, model_ckpt)
