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


def parse_retinaface_line(line):
    parts = line.strip().split()
    if len(parts) != 21:
        return None, None, None

    filename = parts[0]
    x1, y1 = float(parts[1]), float(parts[2])
    w, h = float(parts[3]), float(parts[4])
    x2, y2 = x1 + w, y1 + h

    # 裁剪框
    bbox = [int(max(0, x1)), int(max(0, y1)), int(x2), int(y2)]

    # 5个关键点
    landmarks = np.array([
        [float(parts[5]), float(parts[6])],
        [float(parts[8]), float(parts[9])],
        [float(parts[11]), float(parts[12])],
        [float(parts[14]), float(parts[15])],
        [float(parts[17]), float(parts[18])],
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

    # 平移 landmarks 到局部框坐标
    landmarks_local = landmarks - np.array([x1, y1], dtype=np.float32)

    dst = get_dynamic_template(landmarks_local, output_size=image_size)
    tform = trans.SimilarityTransform()
    tform.estimate(landmarks_local, dst)
    M = tform.params[0:2, :]
    aligned = cv2.warpAffine(face, M, image_size, borderValue=0.0)

    return aligned


def extract_gt_label(filename):
    return "_".join(Path(filename).stem.split("_")[:-1])


def evaluate(images_dir, txt_file, feature_file, model_ckpt, device="cuda"):
    # 初始化模型
    model = get_model('r100', fp16=False).to(device)
    model.load_state_dict(torch.load(model_ckpt))
    model.eval()

    # 特征库
    db = np.load(feature_file, allow_pickle=True)
    db_features = torch.tensor(db["features"]).float().to(device)  # (N, D)
    db_labels = db["labels"]

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    correct = 0
    total = 0

    with open(txt_file, 'r') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Evaluating"):
        filename, bbox, landmarks = parse_retinaface_line(line)
        if filename is None:
            continue

        img_path = Path(images_dir) / filename
        if not img_path.exists():
            print(f"[WARN] Not found: {img_path}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        aligned = align_face(img, bbox, landmarks)
        if aligned is None:
            continue

        image = Image.fromarray(aligned)
        image_tensor = transform(image).unsqueeze(0).to(device)  # (1, 3, 112, 112)

        with torch.no_grad():
            feat = model(image_tensor)
            feat = F.normalize(feat, p=2, dim=1)

        sims = torch.matmul(feat, db_features.T)  # (1, N)
        top1 = torch.argmax(sims, dim=1).item()
        pred_label = db_labels[top1]
        gt_label = extract_gt_label(filename)

        if pred_label == gt_label:
            correct += 1
        else:
            print(f"❌ {filename} | GT: {gt_label} | Pred: {pred_label}")

        total += 1

    acc = correct / total if total > 0 else 0.0
    print(f"\n✅ Total: {total} | Correct: {correct} | Accuracy: {acc:.2%}")



if __name__ == "__main__":
    images_dir = "/scratch/pf2m24/data/Donkey_xiabao_face/images"
    txt_file = "/scratch/pf2m24/data/Donkey_xiabao_face/train_annotations.txt"
    feature_file = "./output_features/features_and_labels_face_train.npz"
    model_ckpt = "/scratch/pf2m24/projects/donkey_place/insightface/recognition/arcface_torch/work_dirs/donkey/model.pt"

    evaluate(images_dir, txt_file, feature_file, model_ckpt, device="cuda" if torch.cuda.is_available() else "cpu")
