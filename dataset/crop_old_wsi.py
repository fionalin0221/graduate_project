import yaml
import os
import shutil
import time
import tifffile as tiff
from tqdm import tqdm
import cv2

config_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'config', 'config_data.yml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
current_computer = config['current_computer']
wsi_type = config['type']
state = config['state']
file_paths = config['computers'][current_computer]['file_paths']

PATCH_SIZE = 448

raw_base_path = file_paths[f'HCC_old_ndpi_path']
save_base_path = file_paths[f'HCC_old_patches_save_path']
# wsis = file_paths[f'HCC_old_wsis']
wsis = [44]

for wsi_id in wsis:
    source_dir = f"{raw_base_path}/Image_{wsi_id}"

    for file_name in os.listdir(source_dir):
        if not file_name.lower().endswith('.tif'):
            continue
        class_name, _ = os.path.splitext(file_name)
        if class_name.startswith("_"):
            class_name = class_name[1:]

        if class_name[0] != "H":
            continue
            
        source_tif = f"{source_dir}/{file_name}"
        # target_dir = f"{save_base_path}/Image_{wsi_id}/{class_name}"
        target_dir = f"{save_base_path}/{wsi_id}/{class_name}"
        # target_tif = f"{target_dir}/{file_name}"
        dims_log = f"{target_dir}/Reshape.txt"

        if not os.path.exists(source_tif):
            print(f"cannot find {source_tif}")
            continue

        os.makedirs(target_dir, exist_ok=True)
        # shutil.move(str(source_tif), str(target_tif))

        try:
            with tiff.TiffFile(source_tif) as tif:
                image_shape = tif.pages[0].shape
                height, width = image_shape[0], image_shape[1]
                
                wq = width // PATCH_SIZE
                hq = height // PATCH_SIZE
                total_patches = wq * hq
                start_time = time.strftime("%Y-%m-%d %H:%M:%S")

                log_info = (
                    f"Time: {start_time}\n"
                    f"File: {source_tif}\n"
                    f"Segmentation Size: {PATCH_SIZE}\n"
                    f"Width: {width} Pixel\n"
                    f"Height: {height} Pixel\n"
                    f"Width Stride: {wq}\n"
                    f"Height Stride: {hq}\n"
                    f"Amount of Patches: {total_patches}\n"
                )
                with open(dims_log, "w", encoding="utf-8") as f:
                    f.write(log_info)
                # print(log_info)

                image_data = tif.asarray(out='memmap')

                for v2 in tqdm(range(hq), desc=f"Cropping {file_name}"):
                    for v1 in range(wq):
                        x = v1 * PATCH_SIZE
                        y = v2 * PATCH_SIZE

                        patch = image_data[y : y + PATCH_SIZE, x : x + PATCH_SIZE]
                        # if patch.mean() > 250:
                        #     continue

                        patch_filename = f"{x}_{y}.tif"
                        # tiff.imwrite(f"{target_dir}/{patch_filename}", patch, compression = 'zlib', compressionargs={'level': 9}) #adobe_deflate #zlib #[cv2.IMWRITE_TIFF_COMPRESSION, 7]
                        tiff.imwrite(
                            f"{target_dir}/{patch_filename}", 
                            patch, 
                            photometric='rgb', # 必須指定色彩空間
                            compression='jpeg', # 改用 jpeg 壓縮
                            compressionargs={'level': 85} # level 這裡代表品質 (0-100)，85 是平衡點
                        )

            print(f"\n[{wsi_id}-{file_name}] already cropped")

        except Exception as e:
            # if os.path.exists(target_tif):
            #     shutil.move(str(target_tif), str(source_tif))
            print(f"\n{file_name}: {e}")
    
        finally:
            # if os.path.exists(target_tif):
            #     shutil.move(str(target_tif), str(source_tif))
            if 'image_data' in locals():
                del image_data
            print(f"Your segmentation has successfully done: {raw_base_path}-{wsi_id}-{class_name}")