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

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
from shapely.geometry import Polygon as ShapePolygon
from shapely.validation import make_valid

def visualize_patches_on_wsi(cancer_type, csv_dir, csv_path, all_region, classes, patch_size=448):
    df = pd.read_csv(csv_path)
    
    color_map = {classes[0]: 'green',}
    color_map[classes[1]] = 'red' if cancer_type == "HCC" else 'blue'
    if len(classes) > 2:
        color_map[classes[2]] = 'yellow'

    fig, ax = plt.subplots(figsize=(12, 12))

    for label, regions in all_region.items():
        if label in classes:
            for region in regions:
                polygon = np.array(region)
                ax.plot(polygon[:, 0], polygon[:, 1], color='black', linewidth=1, alpha=0.8)
                poly_patch = patches.Polygon(polygon, closed=True, fill=True, 
                                             color='gray', alpha=0.1)
                ax.add_patch(poly_patch)

    for _, row in df.iterrows():
        f = row['file_name']
        label = row['label']

        fx, fy = f.replace('.tif', '').split('_')
        x, y = int(fx), int(fy)

        rect = patches.Rectangle((x, y), patch_size, patch_size, 
                                 linewidth=0, 
                                 edgecolor='none', 
                                 facecolor=color_map.get(label, 'yellow'), 
                                 alpha=0.5)
        ax.add_patch(rect)

    ax.set_aspect('equal')
    ax.invert_yaxis() 
    
    plt.title(f"Patch Visualization - {len(df)} patches")
    plt.legend([patches.Patch(color=color_map[c], alpha=0.5) for c in classes], classes)
    
    plt.savefig(f"{csv_dir}/visualization_WSI_{wsi}.png", dpi=300, bbox_inches='tight', facecolor='white')
    # plt.show()

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
    img = cv2.imread(image_path)
    if img is None:
        return 1

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.mean(gray_image) >= 230:  #white
        return 1

    color = ('b','g','r')
    bgr_cal = []
    for i, col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0, 256])
        bgr_cal.append(np.argmax(histr))
    b, g, r = bgr_cal
    if int(b == g == r == 0) == 1:
        return 1
    return 0

def process_image(f, patches_path, all_region, classes, threshold=0.5):

    if check_patch_condition(f"{patches_path}/{f}") == 1:
        return None

    fx, fy = f.split('_')
    fy = fy[:-4]
    left_up = [int(fx), int(fy)]
    right_up = [int(fx) + 448, int(fy)]
    left_down = [int(fx), int(fy) + 448]
    right_down = [int(fx) + 448, int(fy) + 448]
    patch_coords = [left_up, right_up, right_down, left_down]

    # Use corner to decide labeled or not
    # for label_name, regions in all_region.items():
    #     if label_name in classes:
    #         for region in regions:
    #             region = np.array(region)
    #             inside_count = sum(Point_in_Region(pt, region) for pt in patch_coords)
    #             # if all(Point_in_Region(pt, region) for pt in [left_up, right_up, left_down, right_down]):
    #             if inside_count >= 2:
    #                 return f, label_name

    # Use area to decide labeled or not
    patch_poly = ShapePolygon(patch_coords)
    patch_area = patch_poly.area
    for label_name, regions in all_region.items():
        if label_name in classes:
            for region in regions:
                if len(region) < 3:
                    continue
                region_poly = ShapePolygon(region)
                if not region_poly.is_valid:
                    region_poly = make_valid(region_poly)
                    if region_poly.is_empty:
                        continue

                intersection_area = patch_poly.intersection(region_poly).area
                overlap_ratio = intersection_area / patch_area
                
                if overlap_ratio > threshold:
                    return f, label_name

    return None

config_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'config', 'config_data.yml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

current_computer = config['current_computer']
cancer_type = config['type']
state = config['state']
file_paths = config['computers'][current_computer]['file_paths']
class_list = config["class_list"]
classes = [class_list[i] for i in file_paths['classes']]
print(classes)

wsis = file_paths[f'{cancer_type}_wsis']
for wsi in wsis:
    print(f'{cancer_type} WSI : ', wsi)
    if cancer_type == "HCC":
        xml_name = "LIVER_{:05d}.xml".format(wsi)
        # if len(classes) == 2:
        #     xml_name = "LIVER_{:05d}.xml".format(wsi)
        # elif len(classes) == 3:
        #     xml_name = "LIVER_{:05d}_3.xml".format(wsi)
        tree = ET.parse(os.path.join(file_paths[f'HCC_{state}_ndpi_path'], xml_name), parser=ET.XMLParser(encoding="utf-8"))
    elif cancer_type == "CC":
        xml_name = "LIVER_1{:04d}.xml".format(wsi)
        # if len(classes) == 2:
        #     xml_name = "LIVER_1{:04d}.xml".format(wsi)
        # elif len(classes) == 3:
        #     xml_name = "LIVER_1{:04d}_3.xml".format(wsi)
        tree = ET.parse(os.path.join(file_paths['CC_ndpi_path'], xml_name), parser=ET.XMLParser(encoding="utf-8"))
    print(xml_name)
    root = tree.getroot()

    all_region = {}
    for child in root.iter():
        # print(child.tag, child.attrib)
        if child.tag == 'Region':
            current_label = child.attrib['Text']
            if current_label not in all_region:
                all_region[current_label] = []
            all_region[current_label].append([])  # start a new polygon list
        if child.tag == 'Vertex':
            x = int(float(child.attrib['X']))
            y = int(float(child.attrib['Y']))
            all_region[current_label][-1].append((x, y))
    
    if cancer_type == "HCC":
        csv_dir = os.path.join(file_paths['HCC_csv_dir'],f"{wsi+91}")
    elif cancer_type == "CC":
        csv_dir = os.path.join(file_paths['CC_csv_dir'],f"{wsi}")
    os.makedirs(csv_dir, exist_ok=True)

    data_info = {"file_name": [], "label": []}
    nums = [0] * len(classes)

    if cancer_type == "HCC":
        patches_path = os.path.join(file_paths[f'{cancer_type}_{state}_patches_save_path'],f"{wsi}")
    else:
        patches_path = os.path.join(file_paths[f'{cancer_type}_patches_save_path'],f"{wsi}")
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
    if cancer_type == "HCC":
        csv_path = os.path.join(csv_dir, f'{wsi+91}_patch_in_region_filter_{len(classes)}_v2.csv')
    elif cancer_type == "CC":
        csv_path = os.path.join(csv_dir, f'1{wsi:04d}_patch_in_region_filter_{len(classes)}_v2.csv')
    df.to_csv(csv_path, index=False)
    if len(classes) == 2:
        print(nums[0], nums[1])
    elif len(classes) == 3:
        print(nums[0], nums[1], nums[2])
    
    visualize_patches_on_wsi(cancer_type, csv_dir, csv_path, all_region, classes)