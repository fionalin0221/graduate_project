# The path can also be read from a config file, etc.
# OPENSLIDE_PATH = r'C:/openslide-bin-4.0.0.4-windows-x64/bin'

import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import time
import cv2
import traceback
import yaml

config_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'config', 'config_data.yml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
current_computer = config['current_computer']
type = config['type']
state = config['state']
file_paths = config['computers'][current_computer]['file_paths']

if type == "HCC":
    ndpi_path = file_paths[f'{type}_{state}_ndpi_path']
else:
    ndpi_path = file_paths[f'{type}_ndpi_path']
wsis =  file_paths[f'{type}_wsis']

OPENSLIDE_PATH = file_paths['OPENSLIDE_PATH']
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

for wsi in wsis:
    try:
        print(f"Crop {type} WSI-{wsi}...")
        start = time.time()
        if type == "HCC":
            ndpi = f"{ndpi_path}/LIVER_{wsi:05d}.ndpi"
        elif type == "CC":
            ndpi = f"{ndpi_path}/LIVER_1{wsi:04d}.ndpi"
        wsi_openslide = openslide.OpenSlide(ndpi)
        p_w = wsi_openslide.dimensions[0] // 448 + 1
        p_h = wsi_openslide.dimensions[1] // 448 + 1
        
        if type == "HCC":
            patches_save_path = os.path.join(file_paths[f'{type}_{state}_patches_save_path'], f"{wsi}")
        else:
            patches_save_path = os.path.join(file_paths[f'{type}_patches_save_path'], f"{wsi}")

        if not os.path.exists(patches_save_path):
            os.makedirs(patches_save_path)

        print(patches_save_path)

        for height in tqdm(range(0, p_h*448, 448)):
            for width in range(0, p_w*448, 448):
                img = np.array(wsi_openslide.read_region((width, height), 0, (448, 448)))

                im = Image.fromarray(img)
                im = im.convert('RGB')
                im.save(f"{patches_save_path}/{width}_{height}.tif", dpi=(100, 100))
        wsi_openslide.close()
        
        end = time.time()
        print(f"Finish !")
        print(f"Spend : ", end - start)
        print()
    
    except Exception as e:
        print(f"WSI-{wsi} is too large !")
        traceback.print_exc()
        print()