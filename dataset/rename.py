import os
import re

# save_offsets.py
def extract_offsets(csv_path):
    with open(csv_path, "r") as f:
        lines = f.readlines()

    offsets = {}
    current_label = None
    coords = []

    for line in lines:
        line = line.strip()
        if line.startswith("#"):
            if current_label and coords:
                xs, ys = zip(*coords)
                x_min, y_min = int(min(xs)), int(min(ys))
                offsets[current_label] = (x_min, y_min)
            # current_label = line.replace("#", "").strip()
            current_label = re.sub(r'[^\w]', '', line.replace("#", "").strip())
            coords = []
        elif line:
            try:
                x, y = map(float, line.split(","))
                coords.append((x, y))
            except:
                pass
    if current_label and coords:
        xs, ys = zip(*coords)
        x_min, y_min = int(min(xs)), int(min(ys))
        offsets[current_label] = (x_min, y_min)

    # 輸出成 offset.txt
    # output_file = "/workspace/Data/Datas/S17-16817A1_offsets.txt"
    # with open(output_file, "w") as f:
    #     for label, (x, y) in offsets.items():
    #         f.write(f"{label} {x} {y}\n")
    return offsets

# 設定資料夾與 offset
base_dir = "/workspace/Data/Datas/temp/5"
offsets = extract_offsets("/workspace/Data/Datas/18-42041A2.csv")

for label, (x_offset, y_offset) in offsets.items():
    folder = os.path.join(base_dir, label)
    if not os.path.exists(folder):
        print(f"Folder not exist: {folder}")
        continue

    for fname in os.listdir(folder):
        if not fname.endswith(".tif"):
            continue

        match = re.match(rf"{label}-(\d+)-(\d+)-", fname)
        if not match:
            print(f"Cannot extract: {fname}")
            continue

        x_local = int(match.group(1))
        y_local = int(match.group(2))
        x_global = x_offset + x_local
        y_global = y_offset + y_local

        new_name = f"{x_global}_{y_global}.tif"
        src = os.path.join(folder, fname)
        dst = os.path.join(folder, new_name)

        try:
            os.rename(src, dst)
            print(f"{fname} → {new_name}")
        except Exception as e:
            print(f"Cannot rename {fname}: {e}")
