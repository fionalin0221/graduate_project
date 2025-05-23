import pandas as pd
import numpy as np
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import yaml
import time
from concurrent.futures import ThreadPoolExecutor

def check_patch_condition(image_path):
    start_read_img = time.time()
    img = cv2.imread(image_path)
    if img is None:
        return 1
    end_read_img = time.time()
    # print(f"read img time: {end_read_img-start_read_img}")

    # start_color_mean = time.time()
    # blue_mean = np.mean(img[:, :, 2])
    # red_mean = np.mean(img[:, :, 0])
    # if blue_mean <= 200 and red_mean <= 180:
    #     return 1
    # end_color_mean = time.time()
    # print(f"count color mean time: {end_color_mean-start_color_mean}")

    start_mean_pixel = time.time()
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.mean(gray_image) >= 250:  #white
        return 1
    end_mean_pixel = time.time()
    # print(f"count mean pixel time: {end_mean_pixel-start_mean_pixel}")

    start_black_time = time.time()
    color = ('b','g','r')
    bgr_cal = []
    for i, col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0, 256])
        bgr_cal.append(np.argmax(histr))
    b, g, r = bgr_cal
    if int(b == g == r == 0) == 1:
        return 1
    end_black_time = time.time()
    # print(f"end black time: {end_black_time-start_black_time}")
    return 0

def process_files(file_list, folder):
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(check_patch_condition, [os.path.join(folder, f) for f in file_list]), 
                            total=len(file_list), desc="Processing patches", leave=False))
    return [result for result in results if result is not None]

config_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'config', 'config_data.yml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
current_computer = config['current_computer']
type = config['type']
file_paths = config['computers'][current_computer]['file_paths']
state = config['state']
class_list = config["class_list"]
classes = [class_list[i] for i in file_paths['classes']]
print(classes)

random.seed(0)
csv_dir = file_paths['HCC_csv_dir'] if type == "HCC" else file_paths['CC_csv_dir']
if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)

wsis = file_paths[f'{type}_wsis']
# wsis = list(range(1, 92))
for wsi in wsis:
    print(f"{type} WSI-{wsi}")
    if not os.path.exists(os.path.join(csv_dir, str(wsi))):
        os.makedirs(os.path.join(csv_dir, str(wsi)))
    
    if type == "HCC" and state == "old":
        
        cancer_patch_path = f"{file_paths[f'{type}_old_patches_save_path']}/{wsi}/HCC"
        normal_patch_path = f"{file_paths[f'{type}_old_patches_save_path']}/{wsi}/Normal"

        cancer_file_names = [f for f in os.listdir(cancer_patch_path) if f.endswith(".tif")]
        normal_file_names = [f for f in os.listdir(normal_patch_path) if f.endswith(".tif")]

        # cancer_tissue_files = process_files(cancer_file_names, cancer_patch_path)
        # normal_tissue_files = process_files(normal_file_names, normal_patch_path)
        
        # Filter_Region = {cancer_tissue_files + normal_tissue_files}

        # Collect patch paths and labels
        cancer_rows = [
            [fname, classes[1]]
            for fname in cancer_file_names
        ]

        normal_rows = [
            [fname, classes[0]]
            for fname in normal_file_names
        ]

        # Combine and create DataFrame
        df = pd.DataFrame(cancer_rows + normal_rows, columns=["file_name", "label"])

        save_file_name = f"{csv_dir}/{wsi}/{wsi}_patch_in_region_filter_2_v2.csv"
        df.to_csv(save_file_name, index=False)

    else:
        all_patch_path = f"{file_paths[f'{type}_patches_save_path']}/{wsi}"
        all_file_names = [f for f in os.listdir(all_patch_path) if f.endswith(".tif")]

        tissue_files = process_files(all_file_names, all_patch_path)
        Filter_Region = {"file_name": tissue_files}

        save_file_name = (
            f"{wsi+91}/{wsi+91}_all_patches_filter_v2.csv" if (type == "HCC")
            else f"{wsi}/1{wsi:04d}_all_patches_filter_v2.csv"
        )

        pd.DataFrame(Filter_Region).to_csv(f"{csv_dir}/{save_file_name}", index=False)