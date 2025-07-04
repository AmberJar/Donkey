import os
import cv2
import numpy as np
from pathlib import Path
from skimage import transform as trans

# ArcFace标准五点位置（对应112×112）
arcface_template = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)


def parse_line(line):
    """解析一行检测结果，返回bbox和关键点"""
    nums = list(map(float, line.strip().split()))
    if len(nums) != 20:
        return None, None
    x1, y1, x2, y2 = nums[0], nums[1], nums[2], nums[3]
    landmarks = np.array([
        [nums[4], nums[5]],  # left eye
        [nums[7], nums[8]],  # right eye
        [nums[10], nums[11]],  # nose
        [nums[13], nums[14]],  # left mouth
        [nums[16], nums[17]],  # right mouth
    ], dtype=np.float32)
    return (int(x1), int(y1), int(x2), int(y2)), landmarks

def get_dynamic_template(landmarks, output_size=(112, 112), scale=1.2):
    center = np.mean(landmarks, axis=0)
    landmarks_centered = landmarks - center
    max_dist = np.max(np.linalg.norm(landmarks_centered, axis=1))
    scaled = (landmarks_centered / max_dist) * (output_size[0] / 2 / scale) + output_size[0] / 2
    return scaled


def align_face_best(img, bbox, landmarks, image_size=(112, 112)):
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


def align_face(cropped_face, landmarks, image_size=(112, 112)):
    """仿射对齐裁剪后的人脸"""
    tform = trans.SimilarityTransform()
    tform.estimate(landmarks, arcface_template)
    M = tform.params[0:2, :]
    aligned = cv2.warpAffine(cropped_face, M, image_size, borderValue=0.0)
    return aligned

def align_three(img, bbox, landmarks, image_size=(112, 112)):
    x1, y1, x2, y2 = bbox
    face = img[y1:y2, x1:x2]
    if face.shape[0] == 0 or face.shape[1] == 0:
        return None

    # 平移 landmarks 到局部框坐标
    landmarks_local = landmarks[[0, 1, 2], :] - np.array([x1, y1], dtype=np.float32)

    dst = get_dynamic_template(landmarks_local, output_size=image_size)
    selected = landmarks[[0, 1, 2], :]  # 左眼、右眼、鼻尖

    tform = trans.SimilarityTransform()
    tform.estimate(selected, dst)
    M = tform.params[0:2, :]
    aligned = cv2.warpAffine(img, M, image_size, borderValue=0.0)
    return aligned

def process_dataset(txt_path, img_folder, output_root, output_root_, label_output_path):
    img_folder = Path(img_folder)
    output_root = Path(output_root)
    output_root_ = Path(output_root_)
    output_root.mkdir(parents=True, exist_ok=True)

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    label_file = open(label_output_path, "w")
    current_img_name = None
    current_key = None
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        if line.startswith("#"):
            current_img_name = line[1:].strip()
            current_key = current_img_name.rsplit("_", 1)[0]
            i += 1

            # 检查是否仅有一个检测框
            j = i
            num_count = 0
            while j < len(lines) and not lines[j].strip().startswith("#"):
                if lines[j].strip():
                    num_count += 1
                j += 1

            if num_count == 1:
                bbox, landmarks = parse_line(lines[i])
                if bbox is None or landmarks is None:
                    i = j
                    continue

                src_img_path = img_folder / current_img_name
                if not src_img_path.exists():
                    print(f"⚠️ 找不到图像：{src_img_path}")
                    i = j
                    continue

                img = cv2.imread(str(src_img_path))

                if img is None:
                    print(f"⚠️ 图像读取失败：{src_img_path}")
                    i = j
                    continue

                x1, y1, w, h = bbox
                x2 = x1 + w
                y2 = y1 + h

                # 图像尺寸限制（防越界）
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(x2, img.shape[1])
                y2 = min(y2, img.shape[0])
                aligned_best = align_face_best(img,[x1, y1, x2, y2],landmarks,(112,112))
                aligned = align_three(img,[x1, y1, x2, y2], landmarks)

                cropped_face = img[y1:y2, x1:x2]

                # 相对坐标平移
                landmarks = landmarks - np.array([x1, y1], dtype=np.float32)



                class_dir = output_root / current_key
                class_dir.mkdir(parents=True, exist_ok=True)
                dst_img_path = class_dir / current_img_name
                dst_pth = output_root_ / current_key
                pths = dst_pth / current_img_name

                cv2.imwrite(str(dst_img_path), aligned_best)
                cv2.imwrite(str(pths),aligned)
                label_file.write(f"{dst_img_path} {current_key}\n")

            i = j
        else:
            i += 1

    label_file.close()
    print("✅ 图像裁剪 + 对齐完成，label 写入：", label_output_path)


# ======================
# ✅ 主程序入口
# ======================
if __name__ == "__main__":
    # 修改这三个路径
    txt_path = r"E:\BaiduNetdiskDownload\donkeys_all_250604\donkeys_all_250604\train\new_label.txt"  # 标注文件，只含图片名和关键点
    img_folder = r"E:\BaiduNetdiskDownload\donkeys_all_250604\donkeys_all_250604\train\donkey_face"  # 原始图像文件夹
    output_root = r"E:\BaiduNetdiskDownload\donkeys_all_250604\donkeys_all_250604\train\aligment_best"  # 对齐图像保存根目录
    output_root_ = r"E:\BaiduNetdiskDownload\donkeys_all_250604\donkeys_all_250604\train\aligment_three"
    label_output_path = r"E:\BaiduNetdiskDownload\donkeys_all_250604\donkeys_all_250604\train\arcface_label.txt"  # ArcFace label 输出路径

    process_dataset(txt_path, img_folder, output_root, output_root_, label_output_path)
