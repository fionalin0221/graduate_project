import pandas as pd
import numpy as np
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import yaml

def check_patch_condition(image_path):
    color = ('b','g','r')
    img = cv2.imread(image_path)
    # blue_mean = np.mean(img[:, :, 2])
    # red_mean = np.mean(img[:, :, 0])

    ### check partial white ###
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_pixel = np.mean(gray_image)

    bgr_cal = []
    for i, col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0, 256])
        bgr_cal.append(np.argmax(histr))
    b, g, r = bgr_cal

    if b == g == r == 0:
        black = 1
    else:
        black = 0

    if mean_pixel <= 230 and black == 0: #and blue_mean >= 200 and red_mean >= 180:
        return 0
    else:
        return 1

def devide_into_dataset(all_data, valid_num, test_num):
    valid_set = random.sample(all_data, valid_num)
    all_data = [d for d in all_data if d not in valid_set]
    test_set = random.sample(all_data, test_num)
    train_set = [d for d in all_data if d not in test_set]
    return {'train_set': train_set, 'valid_set': valid_set, 'test_set': test_set}

config_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'config', 'config.yml')
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
csv_dir = file_paths['csv_dir']
if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)

wsis = file_paths[f'{type}_wsis']
for wsi in wsis:
    print(f"{type} WSI-{wsi}")
    if not os.path.exists(os.path.join(csv_dir, str(wsi))):
        os.makedirs(os.path.join(csv_dir, str(wsi)))

    file_names = []
    labels  = []
    Filter_Region = {"file_name": [], "label": []}

    if type == "HCC" and state == "old":
        cancer_patch_path = f"{file_paths['old_patches_save_path']}/{wsi}/HCC"
        normal_patch_path = f"{file_paths['old_patches_save_path']}/{wsi}/Normal"
        cancer_file_names = os.listdir(cancer_patch_path)
        normal_file_names = os.listdir(normal_patch_path)
        for idx, file_name in enumerate(tqdm(cancer_file_names)):
            if file_name[-4:] == ".tif":
                if check_patch_condition(f"{cancer_patch_path}/{file_name}") == 0:
                    file_names.append(file_name)
                    labels.append(classes[1])
                    Filter_Region["file_name"].append(file_name)
                    Filter_Region["label"].append(classes[1])

        for idx, file_name in enumerate(tqdm(normal_file_names)):
            if file_name[-4:] == ".tif":
                if check_patch_condition(f"{normal_patch_path}/{file_name}") == 0:
                    file_names.append(file_name)
                    labels.append(classes[0])
                    Filter_Region["file_name"].append(file_name)
                    Filter_Region["label"].append(classes[0])
    else:
        all_patch_path = f"{file_paths['patches_save_path']}/{wsi}"
        all_file_names = os.listdir(all_patch_path)
        for idx, file_name in enumerate(tqdm(all_file_names)):
            if file_name[-4:] == ".tif":
                if check_patch_condition(f"{all_patch_path}/{file_name}") == 0:
                    file_names.append(file_name)
                    # labels.append(all_labels[idx])
                    Filter_Region["file_name"].append(file_name)
                    # Filter_Region["label"].append(all_labels[idx])
    
    file_names = np.array(file_names)
    # labels = np.array(labels)

    save_file_name = (
        f"{wsi}/{wsi}_all_patches_filter_v2.csv" if (type == "HCC" and state == "old")
        else f"{wsi+91}/{wsi+91}_all_patches_filter_v2.csv" if (type == "HCC")
        else f"{wsi}/1{wsi:04d}_all_patches_filter_v2.csv"
    )

    pd.DataFrame(Filter_Region).to_csv(f"{csv_dir}/{save_file_name}", index=False)