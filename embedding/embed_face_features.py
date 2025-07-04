import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm
import timm

def extract_label(filename):
    return "_".join(Path(filename).stem.split("_")[:-1])


def read_yolo_bbox(txt_path, img_shape):
    """
    读取YOLO格式的txt标注，返回像素坐标的bbox（只取第一行）
    """
    if not os.path.exists(txt_path):
        return None
    with open(txt_path, 'r') as f:
        line = f.readline()
        if not line.strip():
            return None
        parts = line.strip().split()
        if len(parts) != 5:
            return None
        _, xc, yc, w, h = map(float, parts)
        H, W = img_shape[:2]
        x1 = int((xc - w / 2) * W)
        y1 = int((yc - h / 2) * H)
        x2 = int((xc + w / 2) * W)
        y2 = int((yc + h / 2) * H)
        return x1, y1, x2, y2

def load_selected_list(path):
    selected = set()
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 提取第一个 .jpg 前缀（支持带空格的文件名）
            tokens = line.split()
            for i, token in enumerate(tokens):
                if token.lower().endswith(".jpg"):
                    # 拼接前 i+1 项作为文件名（支持带空格）
                    filename = " ".join(tokens[:i+1])
                    selected.add(filename)
                    break
    return selected

def build_feature_library(images_dir, txt_dir, save_path, selected_list=None, device="cuda"):
    model = timm.create_model('hf-hub:BVRA/MegaDescriptor-T-224', num_classes=0, pretrained=True)
    model.eval().to(device)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    features = []
    labels = []

    images_dir = Path(images_dir)
    all_images = os.listdir(images_dir)

    # 筛选列表：只保留在 selected_list 中的图像
    if selected_list is not None:
        with open(selected_list, 'r') as f:
            selected_set = load_selected_list(selected_list)
        img_files = [img for img in all_images if img in selected_set]
    else:
        img_files = all_images

    for img_path in tqdm(img_files, desc="Building feature library"):
        txt_path = Path(img_path).with_suffix(".txt")

        img = cv2.imread(str(images_dir / img_path))
        if img is None:
            continue

        bbox = read_yolo_bbox(os.path.join(txt_dir, txt_path), img.shape)
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(x2, img.shape[1]), min(y2, img.shape[0])

        face_crop = img[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        image = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = model(image_tensor)
            feat = F.normalize(feat, p=2, dim=1)

        features.append(feat.cpu().numpy())
        # print(extract_label(img_path))
        labels.append(extract_label(img_path))

    features = np.vstack(features)
    print(f"[INFO] Saving {len(labels)} embeddings to {save_path}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, features=features, labels=labels)


if __name__ == "__main__":
    images_dir = "/scratch/pf2m24/data/Donkey_xiabao_face/images"
    txt_dir = "/scratch/pf2m24/data/Donkey_xiabao_face/donkey_face_yolo_one"
    save_path = "./output_features/features_and_labels_body.npz"
    selected_list = "/scratch/pf2m24/data/Donkey_xiabao_face/embedding.txt"

    build_feature_library(images_dir, txt_dir, save_path, selected_list=selected_list)