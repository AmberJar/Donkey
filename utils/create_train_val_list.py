import os
import random
from pathlib import Path
from tqdm import tqdm

def is_float(s):
    try:
        float(s)
        return True
    except:
        return False

def split_and_generate_annotations_only(
    input_dir,
    output_dir,
    annotation_txt_path,
    train_ratio=0.8,
    seed=42
):
    random.seed(seed)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_filenames = []
    val_filenames = []

    # Step 1: 从目录结构中获取图像文件名
    for class_folder in tqdm(sorted(input_dir.iterdir())):
        if not class_folder.is_dir():
            continue

        images = sorted([
            p for p in class_folder.iterdir()
            if p.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ])

        if len(images) < 2:
            print(f"[Warning] 类别 {class_folder.name} 图像不足2张，跳过")
            continue

        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        if len(images) - split_idx == 0:
            split_idx -= 1

        train_filenames.extend([img.name for img in images[:split_idx]])
        val_filenames.extend([img.name for img in images[split_idx:]])

    # Step 2: 构建 annotation 映射（支持文件名中包含空格）
    annotation_map = {}
    with open(annotation_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                continue
            tokens = line.strip().split()
            if not tokens:
                continue
            for i, token in enumerate(tokens):
                if is_float(token):
                    filename = ' '.join(tokens[:i])
                    annotation_map[filename] = line
                    break

    # Step 3: 写入 train/val annotations
    def write_annotation_subset(filenames, output_file):
        matched = []
        missed = []

        for name in filenames:
            if name in annotation_map:
                matched.append(annotation_map[name])
            else:
                missed.append(name)

        with open(output_file, 'w') as f:
            for line in matched:
                f.write(line + '\n')

        print(f"✅ 写入 {output_file}，共 {len(matched)} 条")
        if missed:
            print(f"⚠️ 找不到对应标注的文件（共 {len(missed)}）：")
            for m in missed[:10]:
                print(f"  - {m}")
            if len(missed) > 10:
                print("  ...")

    write_annotation_subset(train_filenames, output_dir / 'train_annotations.txt')
    write_annotation_subset(val_filenames, output_dir / 'val_annotations.txt')

# 示例调用
if __name__ == '__main__':
    split_and_generate_annotations_only(
        input_dir='/scratch/pf2m24/data/Donkey_xiabao_face/arcface_trainset',
        output_dir='/scratch/pf2m24/data/Donkey_xiabao_face/arcface_testset',
        annotation_txt_path='/scratch/pf2m24/data/Donkey_xiabao_face/all_annotations.txt'
    )
