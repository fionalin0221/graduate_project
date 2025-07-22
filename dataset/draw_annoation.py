import pandas as pd
import matplotlib.pyplot as plt

csv_path = "/workspace/project/S17-16817A1.csv"
# Step 1: 讀入整個檔案為文字行
with open(csv_path, "r") as f:
    lines = f.readlines()

# Step 2: 找到 '# HCC' 區段
hcc_lines = []
in_hcc = False

for line in lines:
    stripped = line.strip()

    if stripped == "# HCC":
        in_hcc = True
        continue
    elif stripped.startswith("#") or stripped == "":
        if in_hcc:
            break  # 已經進入 HCC 而且遇到下一段了
        continue

    if in_hcc:
        hcc_lines.append(stripped)

# Step 3: 解析座標
x_vals = []
y_vals = []

for line in hcc_lines:
    try:
        x, y = map(float, line.split(","))
        x_vals.append(x)
        y_vals.append(y)
    except:
        continue  # 忽略格式錯誤的行

# # Step 4: 封閉輪廓並畫圖
x_vals.append(x_vals[0])
y_vals.append(y_vals[0])

plt.figure(figsize=(6, 6))
plt.plot(x_vals, y_vals, '-o', color='red')
plt.gca().invert_yaxis()  # 顯示方向與影像一致
plt.title("HCC Boundary")
plt.axis("equal")
plt.grid(True)
plt.show()

# import os
# # from PIL import Image
# import tifffile
# import numpy as np
# from tqdm import tqdm
# # Image.MAX_IMAGE_PIXELS = None

# # === 參數設定 ===
# csv_path = "/workspace/project/S17-16817A1.csv"
# patch_size = 448
# output_root = "/workspace/Data/1"

# tif_files = {
#     "HCC": "/workspace/project/TIF/S17-16817 A1_HCC.tif",
#     "Normal": "/workspace/project/TIF/S17-16817 A1_Normal.tif"
# }

# # === Step 1: 解析 CSV 中的標註區塊 ===
# def parse_annotation(csv_path):
#     with open(csv_path, "r") as f:
#         lines = f.readlines()

#     data = {}
#     current_label = None
#     for line in lines:
#         line = line.strip()
#         if line.startswith("#"):
#             current_label = line.replace("#", "").strip()
#             data[current_label] = []
#         elif line == "":
#             continue
#         elif current_label is not None:
#             try:
#                 x, y = map(float, line.split(","))
#                 data[current_label].append((x, y))
#             except:
#                 continue
#     return data

# # === Step 2: 計算外接框 ===
# def get_bounding_box(coords):
#     xs, ys = zip(*coords)
#     x_min = int(min(xs))
#     y_min = int(min(ys))
#     x_max = int(max(xs))
#     y_max = int(max(ys))
#     return x_min, y_min, x_max, y_max

# # === Step 3: 切 patch 並儲存 ===
# def extract_and_save_patches(tif_path, label, offset, output_dir, patch_size):
#     os.makedirs(output_dir, exist_ok=True)
#     img_np = tifffile.imread(tif_path)

#     height, width = img_np.shape[:2]

#     for y_local in tqdm(range(0, height - patch_size + 1, patch_size)):
#         for x_local in range(0, width - patch_size + 1, patch_size):
#             patch = img_np[y_local:y_local + patch_size, x_local:x_local + patch_size]

#             if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
#                 continue  # 防止邊界 patch 小於目標尺寸

#             x_global = offset[0] + x_local
#             y_global = offset[1] + y_local

#             filename = f"{x_global}_{y_global}.tif"
#             tifffile.imwrite(
#                 os.path.join(output_dir, filename),
#                 patch,
#                 compress=0  # 無壓縮，避免 imagecodecs 問題
#             )

# # === 主流程 ===
# annotations = parse_annotation(csv_path)

# for label, coords in annotations.items():
#     if label not in tif_files:
#         print(f"No .tif file of {label}")
#         continue

#     x_min, y_min, x_max, y_max = get_bounding_box(coords)
#     offset = (x_min, y_min)
#     tif_path = tif_files[label]
#     output_dir = os.path.join(output_root, label)

#     print(f"Class: {label}")
#     print(f"Image: {tif_path}")
#     print(f"Offset: {offset}")
#     print(f"Save path: {output_dir}")

#     extract_and_save_patches(
#         tif_path=tif_path,
#         label=label,
#         offset=offset,
#         output_dir=output_dir,
#         patch_size=patch_size
#     )
