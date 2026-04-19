import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import yaml
import cv2
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from shapely.geometry import box
from shapely.strtree import STRtree
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_patches_on_wsi(wsi, classes, csv_dir, csv_path, cancer_type='HCC', patch_size=448):    
    df = pd.read_csv(csv_path)
    
    color_map = {classes[0]: 'green',}
    color_map[classes[1]] = 'red' if cancer_type == "HCC" else 'blue'
    if len(classes) > 2:
        color_map[classes[2]] = 'yellow'

    fig, ax = plt.subplots(figsize=(12, 12))

    for _, row in df.iterrows():
        f = row['file_name']
        label = row['label']

        fx, fy = f.replace('.tif', '').split('_')
        x, y = int(fx), int(fy)

        rect = patches.Rectangle((x, y), patch_size, patch_size, 
                                 linewidth=0, 
                                 edgecolor='none', 
                                 facecolor=color_map.get(label, 'grey'), 
                                 alpha=0.6)
        ax.add_patch(rect)

    # Setting the range of the image
    all_x = df['file_name'].apply(lambda x: int(os.path.splitext(x)[0].split('_')[0]))
    all_y = df['file_name'].apply(lambda x: int(os.path.splitext(x)[0].split('_')[1]))
    
    padding = patch_size * 5
    ax.set_xlim(all_x.min() - padding, all_x.max() + padding)
    ax.set_ylim(all_y.min() - padding, all_y.max() + padding)

    ax.set_aspect('equal')
    ax.invert_yaxis() 
    
    plt.title(f"Patch Visualization - {len(df)} patches")
    plt.legend([patches.Patch(color=color_map[c], alpha=0.5) for c in classes], classes)
    
    plt.savefig(f"{csv_dir}/visualization_WSI_{wsi}_new.png", dpi=300, bbox_inches='tight', facecolor='white')

def check_patch_condition(f, patches_path):
    image_path = f"{patches_path}/{f}"
    img = cv2.imread(image_path)
    if img is None:
        return None

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

def get_geometry_list(patches_path, files, patch_size = 448):
    geometries = []
    valid_files = []
    for f in files:
        try:
            coords = f.replace(".tif", "").split('_')
            x, y = int(coords[0]), int(coords[1])
            poly = box(x, y, x + patch_size, y + patch_size)
            geometries.append(poly)
            valid_files.append(f)
        except Exception:
            continue
    return geometries, valid_files

def filtered(filename, spatial_tree, f_geoms, threshold=0.25, patch_size=448):
    try:
        coords = filename.replace(".tif", "").split('_')
        x, y = int(coords[0]), int(coords[1])
        current_poly = box(x, y, x + patch_size, y + patch_size)

        candidate_indices = spatial_tree.query(current_poly)
        
        for idx in candidate_indices:
            cand_poly = f_geoms[idx]
            intersection_area = current_poly.intersection(cand_poly).area
            if (intersection_area / (patch_size ** 2)) > threshold:
                return True
        return False
    except Exception as e:
        return False

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

    base_path = os.path.join(file_paths[f'HCC_old_patches_save_path'],f"{wsi}")
    if not os.path.exists(base_path):
        print(f"Skipping: {base_path} not found")
        continue

    path_f = os.path.join(base_path, "Fibrosis")
    path_h = os.path.join(base_path, "HCC")
    path_n = os.path.join(base_path, "Normal")

    f_files = [f for f in os.listdir(path_f) if f.endswith(".tif")] if os.path.exists(path_f) else []
    f_geoms, f_valid_names = get_geometry_list(path_f, f_files)
    tree = STRtree(f_geoms) if f_geoms else None

    task_list = [
        (path_f, "F", False),
        (path_h, "H", True),
        (path_n, "N", True)
    ]
    
    for folder, label, need_filter in task_list:
        if not os.path.exists(folder):
            print(f'No {label}')
            continue
        all_files = [f for f in os.listdir(folder) if f.endswith(".tif")]

        with ThreadPoolExecutor(max_workers=8) as executor:
            # futures = {executor.submit(check_patch_condition, f, f"{patches_path}/{d}"): f for f in files}
            futures = []
            for f in all_files:
                if need_filter and tree != None:
                    if filtered(f, tree, f_geoms):
                        continue
                futures.append(executor.submit(check_patch_condition, f, folder))

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {label}"):
                result = future.result()
                if result:
                    filename = result
                    data_info['file_name'].append(filename)
                    data_info['label'].append(label)
                    nums[classes.index(label)] += 1

    csv_path = os.path.join(csv_dir, f'{wsi}_patch_in_region_filter_{len(classes)}_v2.csv') 

    if os.path.exists(csv_path):
        old_csv_path = os.path.join(csv_dir, f'{wsi}_patch_in_region_filter_{len(classes)}_v2_original.csv')
        os.rename(csv_path, old_csv_path)

    df = pd.DataFrame(data_info)
    df.to_csv(csv_path, index=False)
    print([n for n in nums])
    visualize_patches_on_wsi(wsi, classes, csv_dir, csv_path)
