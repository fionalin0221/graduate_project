import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import yaml
import cv2
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def check_patch_condition(f, patches_path):
    image_path = f"{patches_path}/{f}"
    img = cv2.imread(image_path)
    if img is None:
        return None
    # blue_mean = np.mean(img[:, :, 2])
    # red_mean = np.mean(img[:, :, 0])
    # if blue_mean <= 200 and red_mean <= 180:
    #     return 1
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.mean(gray_image) >= 230:  #white
        return None
    color = ('b','g','r')
    bgr_cal = []
    for i, col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0, 256])
        bgr_cal.append(np.argmax(histr))
    b, g, r = bgr_cal
    if int(b == g == r == 0) == 1:
        return None
    return f

config_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'config', 'config_data.yml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

current_computer = config['current_computer']
file_paths = config['computers'][current_computer]['file_paths']
class_list = config["class_list"]
classes = [class_list[i] for i in file_paths['classes']]
print(classes)

wsis = file_paths[f'HCC_old_wsis']
for wsi in wsis:
    print(f'HCC WSI : ', wsi)

    csv_dir = os.path.join(file_paths['HCC_csv_dir'],f"{wsi}")
    os.makedirs(csv_dir, exist_ok=True)

    data_info = {"file_name": [], "label": []}
    nums = [0] * len(classes)

    patches_path = os.path.join(file_paths[f'HCC_old_patches_save_path'],f"{wsi}")

    if not os.path.exists(patches_path):
        print(f"Skipping: {patches_path} not found")
        continue
    dirs = os.listdir(patches_path)
    
    for d in dirs:
        files = os.listdir(f"{patches_path}/{d}")
        label = "H" if d == "HCC" else "N"
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(check_patch_condition, f, f"{patches_path}/{d}"): f for f in files}
            
            for future in tqdm(as_completed(futures), total=len(files), desc="Processing images", leave=False):
                result = future.result()
                if result is not None:
                    filename = result
                    data_info['file_name'].append(filename)
                    data_info['label'].append(label)
                    nums[classes.index(label)] += 1

    # old_dir = os.path.join(csv_dir, 'old')
    # os.makedirs(old_dir, exist_ok=True)

    csv_path = os.path.join(csv_dir, f'{wsi}_patch_in_region_filter_{len(classes)}_v2.csv') 

    # if os.path.exists(csv_path):
    #     old_csv_path = os.path.join(old_dir, os.path.basename(csv_path))
    #     os.rename(csv_path, old_csv_path)

    df = pd.DataFrame(data_info)
    df.to_csv(csv_path, index=False)
    print(nums[0], nums[1])
