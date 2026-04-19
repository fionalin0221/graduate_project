import os
import cv2
import yaml
from tqdm import tqdm
import shutil

config_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'config', 'config_data.yml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

current_computer = config['current_computer']
file_paths = config['computers'][current_computer]['file_paths']
class_list = config["class_list"]
classes = [class_list[i] for i in file_paths['classes']]
print(classes)

# wsis = file_paths[f'HCC_old_wsis']
# wsis = list(range(34, 92))
wsis = [44]
for wsi in wsis:
    print(f'HCC WSI : ', wsi)
    patches_dir = file_paths[f'HCC_old_patches_save_path']
    src_dir = f"{patches_dir}/Image_{wsi}"
    dst_dir = f"{patches_dir}/{wsi}/Fibrosis"
    os.makedirs(dst_dir, exist_ok=True)

    for d in os.listdir(src_dir):
        if d[0] != 'F':
            continue
        print(d)
        for f in tqdm(os.listdir(f'{src_dir}/{d}')):
            if not f.endswith(".tif"):
                continue
            image_path = f'{src_dir}/{d}/{f}'
            img = cv2.imread(image_path)
            if img.mean() <= 250:
                src = f'{src_dir}/{d}/{f}'
                dst = f'{dst_dir}/{f}'
                shutil.copy2(src, dst)