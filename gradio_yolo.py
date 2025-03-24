import torch
import gradio as gr
from PIL import Image
from ultralytics import YOLO
import timm
import torchvision.transforms as T
import numpy as np
import os
import torch.nn.functional as F
from collections import defaultdict


def yolo_to_pixel(bbox, img_width, img_height):
    x_center, y_center, w, h = bbox

    # 反归一化为像素坐标
    x_center *= img_width
    y_center *= img_height
    w *= img_width
    h *= img_height

    # 计算像素坐标边界框 (x_min, y_min, x_max, y_max)
    x_min = x_center - w / 2
    y_min = y_center - h / 2
    x_max = x_center + w / 2
    y_max = y_center + h / 2

    # 确保坐标在图像范围内
    x_min = max(0, min(img_width, round(x_min)))
    y_min = max(0, min(img_height, round(y_min)))
    x_max = max(0, min(img_width, round(x_max)))
    y_max = max(0, min(img_height, round(y_max)))

    return x_min, y_min, x_max, y_max

def matching(image, detection_results):
    width, height = image.size

    annotated_results = []
    for result in detection_results:
        boxes = result.boxes  # 获取所有边界框
        for box in boxes:
            # 获取边界框的坐标
            bbox = box.xyxy[0].tolist()  # 转换为 [x_min, y_min, x_max, y_max] 格式
            confidence = box.conf[0].item()  # 置信度
            class_id = int(box.cls[0].item())  # 类别ID
            print(f"BBox: {bbox}, Confidence: {confidence}, Class ID: {class_id}")
    
            x_min, y_min, x_max, y_max = yolo_to_pixel(bbox, width, height)

            # 裁剪原图中的边界框区域
            cropped_image = image.crop((x_min, y_min, x_max, y_max))
       
            # 应用图像变换
            query = MatchingModel(transform(cropped_image).unsqueeze(0))
            sim_matrix = F.cosine_similarity(query, torch.tensor(features))

            scores, idx = sim_matrix.topk(k=1, dim=0)
            idx = idx.cpu().numpy()[0]
            donkey_name = merged_list[idx]

            donkey_name = donkey_name.split('_')[0]
            
            # 保存边界框和识别结果
            annotated_results.append({
                "bbox": (x_min, y_min, x_max, y_max),
                "name": donkey_name,
                "confidence": confidence,
                "class_id": class_id
            })
            
    return annotated_results


def update_yolo_results_with_names(results, annotated_results):
    """
    根据 annotated_results 更新 YOLO 的 results 中的类别名称 (name)。

    :param results: YOLO 模型的预测结果对象
    :param annotated_results: 匹配函数返回的边界框和名字信息
    :return: 更新后的 YOLO 结果对象
    """
    for i, annotation in enumerate(annotated_results):
        if i < len(results[0].boxes):  # 确保索引不越界
            class_name = annotation["name"]
            results[0].names[i] = class_name
    return results

def predict_image(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')

    results = model.predict(source=img, conf=0.25)
    annotated_results = matching(img, results)
    print('===================================')
    print(annotated_results)
    print('===================================')
    results = update_yolo_results_with_names(results, annotated_results)
    im_array = results[0].plot()
    
    pil_img = Image.fromarray(im_array[..., ::-1])

    return pil_img


if __name__ == '__main__':
    # model init
    model = YOLO('/data/Jar/projects/ultralytics/ultralytics/runs/detect/yolov8_large/weights/best.pt')

    name = 'hf-hub:BVRA/MegaDescriptor-T-224'
    MatchingModel = timm.create_model(name, num_classes=0, pretrained=True)

    # 图像处理
    transform = T.Compose([T.Resize([224, 224]), 
                       T.ToTensor(), 
                       T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    # 加载library
    data = np.load('/data/Jar/projects/ultralytics/features_and_labels.npz')
    features = data['features']
    merged_list = data['labels']

    #res = predict_image('/data/Elio/data/Donkeys/trainset/images/train/Amber_00002.jpg')

    # 创建界面
    iface = gr.Interface(
        fn=predict_image,
        inputs=gr.Image(type='pil'),
        outputs='image',
        examples=['./examples/Donkeys/Amber_00002.jpg', 
                  './examples/Donkeys/Moses_00001.jpg', 
                  './examples/Donkeys/Daphne_00005.jpg'],
        title='Donkey Detection and Identification',
        description='Version 0.1',
    )

    # 启动接口，指定服务器地址和端口号
    iface.launch(
        share=True,                # 不生成公网分享链接
        server_name='0.0.0.0',      # 设置为 0.0.0.0 允许外部访问
        server_port=30044           # 指定端口号
    )