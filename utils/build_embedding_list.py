from pathlib import Path

# === 配置路径 ===
input_txt_path = 'your_input.txt'         # 标注文件路径
classes_path = 'classes.txt'              # 类别列表路径（类名用下划线）
output_txt_path = 'filtered_output.txt'   # 输出路径

# === 读取 classes.txt ===
with open(classes_path, 'r') as f:
    valid_classes = set(line.strip() for line in f if line.strip())

# === 读取输入标注文本 ===
with open(input_txt_path, 'r') as f:
    lines = f.readlines()

# === 筛选逻辑 ===
filtered = []
i = 0
while i < len(lines):
    line = lines[i].strip()
    if line.startswith('#'):
        original_name = line[1:].strip()  # e.g., "Murphy_110flip.jpg"
        # 提取类名前缀（遇到第一个下划线为止），再将空格替换为下划线与 classes.txt 对齐
        class_name = original_name.split('_')[0].replace(' ', '_')

        if class_name in valid_classes:
            if i + 1 < len(lines) and not lines[i + 1].startswith('#'):
                data_line = lines[i + 1].strip()
                filtered.append(f"{original_name} {data_line}")
                i += 1  # 跳过数据行
    i += 1

# === 写入结果 ===
with open(output_txt_path, 'w') as f:
    f.write('\n'.join(filtered))

print(f"完成！共筛选出 {len(filtered)} 条，保存在 {output_txt_path}")