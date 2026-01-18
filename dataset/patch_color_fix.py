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

# Ubuntu CC
# wsis = [97, 98, 99, 100, 108, 109, 110, 111, 122, 123, 124, 130, 131, 134, 136, 137, 138, 143, 144, 146, 163, 170, 171, 173, 174, 175, 177, 178, 179, 180, 183, 184, 185, 190, 191, 192, 201, 202, 204, 206, 207, 208, 291, 376, 374, 375, 377, 189, 297, 298, 299, 300, 325, 328, 329, 330, 331, 363, 373, 378, 379, 380, 391, 392, 400, 401, 402, 390, 472, 473, 474, 475, 476, 483, 484, 410, 422, 454, 455, 459, 460, 461, 469, 470, 471, 468]
# docker HCC
# wsis = [62, 63, 69, 78, 79, 80, 87, 165, 166, 168, 169] #26, 42, 60, 
# wsis = [97, 98, 99, 116, 136, 146, 147, 148, 158, 163, 164, 165, 264, 363, 373, 376, 377, 378, 379, 380, 390, 391, 392, 400, 401, 402, 406, 407, 408, 409, 410, 422, 454, 455, 459, 460, 461, 469, 470, 471, 472, 473, 474, 475, 476, 483, 484, 487]
# docker CC
wsis = [88, 100, 108, 109, 110, 111, 118, 122, 123, 124, 130, 131, 134, 135, 136, 137, 138, 143, 144, 145, 167, 168, 169, 170, 171, 173, 174, 175, 177, 178, 179, 180, 183, 184, 185, 189, 190, 191, 192, 201, 202, 204, 206, 207, 208, 222, 291, 297, 298, 299, 300, 325, 328, 374, 375]
# wsis = [244, 246, 247, 248, 249, 250, 263, 265, 266, 269, 275, 276, 277, 296, 245]

for wsi in wsis:
    print(f"Processing {wsi}...")
    # input_dir = f"/media/ipmclab-2/HDD8T/CC_Patch/{wsi}"
    input_dir = f"/workspace/Data/Datas/CC_Patch/{wsi}"
    if not os.path.isdir(input_dir):
        print(f"{input_dir} is not exist.")
        continue
    all_tif_paths = [
        os.path.join(input_dir, filename)
        for filename in os.listdir(input_dir)
        if filename.endswith(".tif")
    ]

    with ProcessPoolExecutor() as executor:
        for result in tqdm(executor.map(fix_color_and_save, all_tif_paths), total=len(all_tif_paths)):
            pass
