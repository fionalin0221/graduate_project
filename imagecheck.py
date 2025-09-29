from PIL import Image
import os
from tqdm import tqdm

# base_path = "/workspace/Data/Datas/CC_Patch"
base_path = "/media/ipmclab-2/HDD8T/CC_Patch"
wsis = [406, 407, 408, 410, 422, 454, 455, 459, 460, 461, 470, 471, 474, 475, 476, 483, 484]

with open("error_files.txt", "w") as fout:
    for wsi in wsis:
        folder = f"{base_path}/{wsi}"

        for f in tqdm(os.listdir(folder), leave=False):
            if f.lower().endswith((".tif", ".tiff")):
                path = os.path.join(folder, f)
                try:
                    with Image.open(path) as img:
                        # img.verify()
                        img.getbbox()  # just force minimal access
                except Exception as e:
                    fout.write(path + "\n")
                    fout.flush()  # ensure it's written immediately
                    tqdm.write(path)
