import os
import cv2
import numpy as np
from pathlib import Path
from skimage import transform as trans

# ArcFace 标准五点位置（用于可选替代）
arcface_template = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)


def parse_line(line):
    """解析检测框 + 5个关键点"""
    nums = list(map(float, line.strip().split()))
    if len(nums) != 20:
        return None, None
    x1, y1, w, h = nums[0], nums[1], nums[2], nums[3]
    bbox = [x1, y1, w, h]
    landmarks = np.array([
        [nums[4], nums[5]],
        [nums[7], nums[8]],
        [nums[10], nums[11]],
        [nums[13], nums[14]],
        [nums[16], nums[17]],
    ], dtype=np.float32)
    return bbox, landmarks


def get_dynamic_template(landmarks, output_size=(112, 112), scale=1.2):
    """生成自适应对齐模板"""
    center = np.mean(landmarks, axis=0)
    landmarks_centered = landmarks - center
    max_dist = np.max(np.linalg.norm(landmarks_centered, axis=1))
    scaled = (landmarks_centered / max_dist) * (output_size[0] / 2 / scale) + output_size[0] / 2
    return scaled


def align_face(face_img, landmarks, image_size=(112, 112)):
    dst = get_dynamic_template(landmarks, output_size=image_size)
    tform = trans.SimilarityTransform()
    tform.estimate(landmarks, dst)
    M = tform.params[0:2, :]
    aligned = cv2.warpAffine(face_img, M, image_size, borderValue=0.0)
    return aligned


def extract_label_key(filename):
    return filename.rsplit("_", 1)[0]


def process_dataset(txt_path, img_folder, output_root, label_output_path):
    img_folder = Path(img_folder)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    label_file = open(label_output_path, "w")
    with open(txt_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("#"):
            filename = line[1:].strip()
            class_key = extract_label_key(filename)

            # 检查是否只有一个检测框（跳过多个）
            j = i + 1
            valid_lines = []
            while j < len(lines) and not lines[j].strip().startswith("#"):
                if lines[j].strip():
                    valid_lines.append(lines[j])
                j += 1

            if len(valid_lines) != 1:
                i = j
                continue

            bbox, landmarks = parse_line(valid_lines[0])
            if bbox is None or landmarks is None:
                i = j
                continue

            img_path = img_folder / filename
            if not img_path.exists():
                print(f"❌ 找不到图像: {img_path}")
                i = j
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"❌ 图像读取失败: {img_path}")
                i = j
                continue

            x1, y1, w, h = bbox
            x2, y2 = int(x1 + w), int(y1 + h)
            x1, y1 = int(max(0, x1)), int(max(0, y1))
            x2, y2 = min(x2, img.shape[1]), min(y2, img.shape[0])
            cropped = img[y1:y2, x1:x2]

            if cropped.shape[0] < 20 or cropped.shape[1] < 20:
                print(f"⚠️ 裁剪尺寸异常: {filename}")
                i = j
                continue

            landmarks -= np.array([x1, y1], dtype=np.float32)
            aligned = align_face(cropped, landmarks)

            class_dir = output_root / class_key
            class_dir.mkdir(parents=True, exist_ok=True)
            save_path = class_dir / filename
            cv2.imwrite(str(save_path), aligned)
            label_file.write(f"{save_path} {class_key}\n")

            i = j
        else:
            i += 1

    label_file.close()
    print(f"✅ 处理完成。图像保存在 {output_root}，标签写入 {label_output_path}")


# ========== 脚本入口 ==========
if __name__ == "__main__":
    txt_path = "/scratch/pf2m24/data/Donkey_xiabao_face/label.txt"
    img_folder = "/scratch/pf2m24/data/Donkey_xiabao_face/images"
    output_root = "/scratch/pf2m24/data/Donkey_xiabao_face/arcface_trainset"
    label_output_path = "/scratch/pf2m24/data/Donkey_xiabao_face/arcface_label.txt"

    process_dataset(txt_path, img_folder, output_root, label_output_path)
