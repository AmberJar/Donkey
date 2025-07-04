import json
import os

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def point_in_box(point, box):
    x, y = point
    (x1, y1), (x2, y2) = box
    return x1 <= x <= x2 and y1 <= y <= y2

def process_pair(a_path, b_path):
    data_a = load_json(a_path)
    data_b = load_json(b_path)

    rectangles = [shape for shape in data_a['shapes'] if shape['shape_type'] == 'rectangle']

    keypoint_labels = ['left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth']
    keypoints = [shape['points'][0] for shape in data_a['shapes']
                 if shape['shape_type'] == 'point' and shape['label'] in keypoint_labels]

    if len(keypoints) < 5:
        return False  # 点不全，跳过

    for rect in rectangles:
        rect_box = rect['points']
        if all(point_in_box(pt, rect_box) for pt in keypoints):
            # 匹配成功，加入 bbox 到 B
            new_box = {
                "label": "donkey",
                "points": rect_box,
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {},
                "mask": None
            }
            data_b['shapes'].append(new_box)
            save_json(data_b, b_path)
            return True

    return False  # 没有找到匹配的 bbox

def process_folder(a_dir, b_dir, output_file="unmatched_files.txt"):
    unmatched = []
    for file_name in os.listdir(a_dir):
        if not file_name.endswith(".json"):
            continue
        a_path = os.path.join(a_dir, file_name)
        b_path = os.path.join(b_dir, file_name)

        if not os.path.exists(b_path):
            print(f"⚠️ 未找到对应 B 文件：{file_name}")
            continue

        matched = process_pair(a_path, b_path)
        if not matched:
            unmatched.append(file_name)

    if unmatched:
        with open(output_file, 'w') as f:
            for name in unmatched:
                f.write(name + '\n')
        print(f"❌ 有 {len(unmatched)} 个文件未匹配成功，见 {output_file}")
    else:
        print("✅ 所有文件都匹配成功")

# 示例用法（替换路径）
a_json_dir = r"E:\download\Meta-CSKT-main\Meta-CSKT-main\labelme_jsons"
b_json_dir = r"E:\BaiduNetdiskDownload\donkey_face_json"
process_folder(a_json_dir, b_json_dir)
