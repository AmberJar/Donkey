import json
import os

# 设置输入输出路径
# json_dir = r"path\to\your\jsons"   # 你的标注json路径
# image_root = "donkey_face"         # 图片路径前缀
# output_txt = "retinaface_format.txt"
import sys


def point_in_box(point, box):
    x, y = point
    (x1, y1), (x2, y2) = box
    return x1 <= x <= x2 and y1 <= y <= y2

def find_points_in_box(shapes, box):
    point_dict = {
        "left_eye": [-1.0, -1.0, -1],
        "right_eye": [-1.0, -1.0, -1],
        "nose": [-1.0, -1.0, -1],
        "left_mouth": [-1.0, -1.0, -1],
        "right_mouth": [-1.0, -1.0, -1],
    }

    for shape in shapes:
        if shape["shape_type"] == "point":
            label = shape["label"]
            if label in point_dict:
                pt = shape["points"][0]
                if point_in_box(pt, box):
                    point_dict[label] = [pt[0], pt[1], 1]

    return point_dict

def convert_json(json_path):
    lines = []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_path = json_path.split("\\")[-1].split('.')[0]+'.jpg'

    all_shapes = data["shapes"]
    rects = [s for s in all_shapes if s["shape_type"] == "rectangle"]

    for rect in rects:
        (x1, y1), (x2, y2) = rect["points"]
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        w, h = x2 - x1, y2 - y1

        # 找出该框内的五个点
        pts = find_points_in_box(all_shapes, ((x1, y1), (x2, y2)))

        # 构造一行
        line = f"{int(x1)} {int(y1)} {int(w)} {int(h)}"
        for key in ["left_eye", "right_eye", "nose", "left_mouth", "right_mouth"]:
            x, y, v = pts[key]
            line += f" {x:.3f} {y:.3f} {v}"
        line += " 1.0"  # score

        lines.append((image_path, line))
    return lines

def convert_all(json_dir, output_txt):
    with open(output_txt, 'w', encoding='utf-8') as out_f:
        for file in sorted(os.listdir(json_dir)):
            if not file.endswith(".json"):
                continue
            json_path = os.path.join(json_dir, file)
            try:
                lines = convert_json(json_path)
                if lines:
                    out_f.write(f"# {lines[0][0]}\n")
                    for _, line in lines:
                        out_f.write(line + "\n")
            except Exception as e:
                print(f"[!] Error processing {file}: {e}")

    print(f"\n✅ 转换完成，输出文件：{output_txt}")

# 运行
convert_all(r"D:\wechat_files\WeChat Files\wxid_0806318062911\FileStorage\File\2025-06\donkey\donkey_imp_label", r"D:\wechat_files\WeChat Files\wxid_0806318062911\FileStorage\File\2025-06\donkey\new_label.txt")
