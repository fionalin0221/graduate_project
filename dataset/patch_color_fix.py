import cv2
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def fix_color_and_save(image_path):
    try:
        img_bgr_wrong = cv2.imread(image_path)
        if img_bgr_wrong is None:
            return f"Failed to read: {image_path}"
        img_rgb = cv2.cvtColor(img_bgr_wrong, cv2.COLOR_BGR2RGB)
        cv2.imwrite(image_path, img_rgb)
        return f"Fixed: {image_path}"
    except Exception as e:
        return f"Error processing {image_path}: {str(e)}"

# wsis = [97, 98, 99, 100, 108, 109, 110, 111, 116, 122, 123, 124, 130, 131, 134, 136, 137, 138, 143, 144, 147, 148, 158, 244, 245, 246, 247, 248, 249, 250, 291, 374, 300, 297, 375, 376, 377, 169, 170, 171, 175, 178, 183, 184, 185, 208]
# 77, 107, 57, 68, 89, 90, 92, 95, 98, 99, 103, 104,  109, 111, 120, 121, 122, 129, 131, 132, 135, 139, 141, 142, 145, 146, 149, 150, 153, 156, 159, 161, 162, 164,
wsis = [165, 166, 168, 169, 60, 62, 63, 69, 78, 79, 80, 87]

for wsi in wsis:
    print(f"Processing {wsi}...")
    input_dir = f"/workspace/Data/Datas/HCC_Patch/{wsi}"
    if not os.path.isdir(input_dir):
        print(f"⚠️ {input_dir} is not exist.")
        continue
    all_tif_paths = [
        os.path.join(input_dir, filename)
        for filename in os.listdir(input_dir)
        if filename.endswith(".tif")
    ]

    with ProcessPoolExecutor() as executor:
        for result in tqdm(executor.map(fix_color_and_save, all_tif_paths), total=len(all_tif_paths)):
            pass
