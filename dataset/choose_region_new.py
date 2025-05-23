import xml.etree.ElementTree as ET
import numpy as np
import os
from shutil import copy
from tqdm import tqdm
import pandas as pd
import matplotlib.path as mpath
import yaml
import cv2
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def Ray_Segment(point, s_point, e_point):
    if s_point[1] == e_point[1]:
        return False
    if s_point[1] > point[1] and e_point[1] > point[1]:
        return False
    if s_point[1] < point[1] and e_point[1] < point[1]:
        return False
    if s_point[1] == point[1] and e_point[1] > point[1]:
        return False
    if e_point[1] == point[1] and s_point[1] > point[1]:
        return False
    if s_point[0] < point[0] and e_point[1] < point[1]:
        return False
    
    xseg = e_point[0]-(e_point[0]-s_point[0])*(e_point[1]-point[1])/(e_point[1]-s_point[1])
    
    if xseg < point[0]:
        return False
    
    return True


def Point_in_Region(point, region):
    xmin = np.min(region[:, 0])
    ymin = np.min(region[:, 1])
    xmax = np.max(region[:, 0])
    ymax = np.max(region[:, 1])
    if (point[0] > xmax or point[0] < xmin or point[1] > ymax or point[1] < ymin):
        return False
    
    count = 0
    for i in range(len(region)-1):
        if Ray_Segment(point, region[i], region[i+1]) == True:
            count += 1

    if count % 2 == 1:
        return True
    else:
        return False

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

def process_image(f, patches_path, all_region, classes):

    if check_patch_condition(f"{patches_path}/{f}") == 1:
        return None

    fx, fy = f.split('_')
    fy = fy[:-4]
    left_up = [int(fx), int(fy)]
    right_up = [int(fx) + 448, int(fy)]
    left_down = [int(fx), int(fy) + 448]
    right_down = [int(fx) + 448, int(fy) + 448]

    for k, v in all_region.items():
        if k[0] in classes:
            region = np.array(v)
            if all(Point_in_Region(pt, region) for pt in [left_up, right_up, left_down, right_down]):
                return f, k[0]

    return None

config_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'config', 'config_data.yml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

current_computer = config['current_computer']
type = config['type']
file_paths = config['computers'][current_computer]['file_paths']
class_list = config["class_list"]
classes = [class_list[i] for i in file_paths['classes']]
print(classes)

wsis = file_paths[f'{type}_wsis']
for wsi in wsis:
    print(f'{type} WSI : ', wsi)
    if type == "HCC":
        xml_name = "LIVER_{:05d}.xml".format(wsi)
        tree = ET.parse(os.path.join(file_paths['HCC_ndpi_path'], xml_name), parser=ET.XMLParser(encoding="utf-8"))
    elif type == "CC":
        xml_name = "LIVER_1{:04d}.xml".format(wsi)
        tree = ET.parse(os.path.join(file_paths['CC_ndpi_path'], xml_name), parser=ET.XMLParser(encoding="utf-8"))
    root = tree.getroot()

    all_region = {}
    for child in root.iter():
        # print(child.tag, child.attrib)
        if child.tag == 'Region':
            label_id = child.attrib['Text'] + child.attrib['Id']
            all_region[label_id] = []
        if child.tag == 'Vertex':
            x = int(float(child.attrib['X']))
            y = int(float(child.attrib['Y']))
            all_region[label_id].append((x, y))
    if type == "HCC":
        csv_dir = os.path.join(file_paths['HCC_csv_dir'],f"{wsi+91}")
    elif type == "CC":
        csv_dir = os.path.join(file_paths['CC_csv_dir'],f"{wsi}")
    os.makedirs(csv_dir, exist_ok=True)

    data_info = {"file_name": [], "label": []}
    nums = [0] * len(classes)

    patches_path = os.path.join(file_paths[f'{type}_patches_save_path'],f"{wsi}")
    if not os.path.exists(patches_path):
        print(f"Skipping: {patches_path} not found")
        continue
    dir_files = os.listdir(patches_path)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_image, f, patches_path, all_region, classes): f for f in dir_files}
        
        for future in tqdm(as_completed(futures), total=len(dir_files), desc="Processing images", leave=False):
            result = future.result()
            if result is not None:
                filename, label = result
                data_info['file_name'].append(filename)
                data_info['label'].append(label)
                nums[classes.index(label)] += 1


    df = pd.DataFrame(data_info)
    if type == "HCC":
        df.to_csv(os.path.join(csv_dir, f'{wsi+91}_patch_in_region_filter_{len(classes)}_v2.csv'), index=False)
    elif type == "CC":
        df.to_csv(os.path.join(csv_dir, f'1{wsi:04d}_patch_in_region_filter_{len(classes)}_v2.csv'), index=False)
    print(nums[0], nums[1])
