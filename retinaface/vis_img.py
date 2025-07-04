import os
import cv2

# 路径配置
img_root = r"E:\BaiduNetdiskDownload\donkeys_all_250604\donkeys_all_250604\train\donkey_face"
label_txt = r"E:\BaiduNetdiskDownload\donkeys_all_250604\donkeys_all_250604\train\new_label.txt"
save_dir = r"E:\BaiduNetdiskDownload\widerface\widerface\train\vis"
os.makedirs(save_dir, exist_ok=True)

# 读取 label.txt
with open(label_txt, 'r') as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    line = lines[i].strip()

    # 处理注释行：提取图片名
    if line.startswith("#"):
        filename = line[1:].strip()
        i += 1
        if i >= len(lines): break

        data_line = lines[i].strip()
        parts = list(map(float, data_line.split()))
        if len(parts) < 20:
            print(f"[×] Invalid annotation: {filename}")
            i += 1
            continue

        # 读取图像
        img_path = os.path.join(img_root, filename)
        if not os.path.exists(img_path):
            print(f"[×] Image not found: {img_path}")
            i += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[×] Cannot read image: {img_path}")
            i += 1
            continue

        # 画框
        x1, y1, x2, y2 = map(int, parts[:4])
        x2 = x2+x1
        y2 = y2+y1
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 画关键点（跳过每个点后的v）
        kps = []
        for j in range(5):
            kp_x = int(parts[4 + j * 3])
            kp_y = int(parts[5 + j * 3])
            visibility = parts[6 + j * 3]
            if kp_x >= 0 and kp_y >= 0 and visibility > 0:
                kps.append((kp_x, kp_y))

        for (x, y) in kps:
            cv2.circle(img, (x, y), 12, (0, 0, 255), -1)

        # 保存图像
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, img)
        print(f"[✓] Saved: {save_path}")

    i += 1
