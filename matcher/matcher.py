import torch
import numpy as np
import torch.nn.functional as F

from PIL import Image
import torchvision.transforms as T
import timm
from insightface.recognition.arcface_torch.backbones import get_model
import cv2
from skimage import transform as trans


def get_dynamic_template(landmarks, output_size=(112, 112), scale=1.2):
    """
    根据原始 landmarks 和设定比例生成动态模板
    """
    center = np.mean(landmarks, axis=0)
    landmarks_centered = landmarks - center

    max_dist = np.max(np.linalg.norm(landmarks_centered, axis=1))
    scaled_landmarks = (landmarks_centered / max_dist) * (output_size[0] / 2 / scale) + output_size[0] / 2

    return scaled_landmarks


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


def align_face(cropped_face, landmarks, image_size=(112, 112)):
    """仿射对齐裁剪后的人脸"""
    tform = trans.SimilarityTransform()
    dst_points = get_dynamic_template(landmarks)
    tform.estimate(landmarks, dst_points)
    # tform.estimate(landmarks, arcface_template)
    M = tform.params[0:2, :]
    aligned = cv2.warpAffine(cropped_face, M, image_size, borderValue=0.0)
    return aligned


class DonkeyFaceMatcher:
    def __init__(self, model, feature_file: str, device: str = "cuda"):
        self.model = model.to(device).eval()
        self.device = device
        self.feature_file = feature_file

        # 加载预生成的特征库
        data = np.load(feature_file, allow_pickle=True)
        self.library_features = torch.tensor(data["features"]).float().to(device)  # [N, D]
        self.library_labels = data["labels"]  # list of labels, e.g. ['A', 'B', ...]

    def extract_feature(self, image_tensor: torch.Tensor):
        """
        输入一张图像 (1, 3, H, W)，输出特征 (1, D)
        """
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            feature = self.model(image_tensor)
            feature = torch.nn.functional.normalize(feature, p=2, dim=1)
        return feature  # (1, D)

    def match(self, query_feature: torch.Tensor, topk=1):
        """
        query_feature: (1, D)
        返回 topk 个最相似的 label 和相似度
        """
        # query_feature = F.normalize(query_feature, dim=-1)
        # library_feature = F.normalize(self.library_features, dim=-1)
        print(query_feature.shape)
        print(self.library_features.shape)
        similarity = torch.matmul(query_feature, self.library_features.T)  # (1, N)
        topk_sim, topk_indices = torch.topk(similarity, k=topk, dim=1)

        topk_labels = [self.library_labels[i] for i in topk_indices[0].cpu().numpy()]
        topk_sim_scores = topk_sim[0].cpu().numpy().tolist()
        return list(zip(topk_labels, topk_sim_scores))


if __name__ == '__main__':
    # 加载图像
    img_path = "/scratch/pf2m24/data/Donkey_xiabao_face/images/Angel_00012.jpg"
    # image = Image.open(img_path).convert('RGB')
    img = cv2.imread(str(img_path))
    txt_input = "1872.001 1190.999 384.000 497.000 2044.194 1448.065 1.000 1960.323 1435.161 1.000 1924.839 1596.452 1.000 2008.710 1609.355 1.000 1953.871 1660.968 1.000 1.000"

    # 仿射变换
    bbox, landmarks = parse_line(txt_input)
    x1, y1, w, h = bbox
    x2 = x1 + w
    y2 = y1 + h

    # 图像尺寸限制（防越界）
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(x2, img.shape[1])
    y2 = min(y2, img.shape[0])

    cropped_face = img[y1:y2, x1:x2]

    landmarks = landmarks - np.array([x1, y1], dtype=np.float32)

    aligned = align_face(cropped_face, landmarks)
    cv2.imwrite('/scratch/pf2m24/projects/donkey_place/outputs/aligned.jpg', aligned)

    # 图像预处理
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    image = Image.fromarray(aligned)
    image_tensor = transform(image).unsqueeze(0)  # (1, 3, H, W)

    # 创建模型和匹配器
    # model = timm.create_model("hf-hub:BVRA/MegaDescriptor-T-224", num_classes=0, pretrained=True)
    model = get_model('r100', fp16=False)
    model.load_state_dict(torch.load(
        '/scratch/pf2m24/projects/donkey_place/insightface/recognition/arcface_torch/work_dirs/donkey/model.pt'))
    model.eval()
    matcher = DonkeyFaceMatcher(model, "./output_features/features_and_labels_face.npz", device="cuda")

    # 匹配结果
    query_feat = matcher.extract_feature(image_tensor)
    top_result = matcher.match(query_feat, topk=1)
    print(f"预测结果：{top_result[0][0]}（相似度: {top_result[0][1]:.3f}）")
