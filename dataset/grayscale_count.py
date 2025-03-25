import os
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm

CC_csv_dir = "/workspace/Data/Results/CC_NDPI/Data_Info"
CC_patches_save_path = "/workspace/Data/Datas/CC_Patch"
output_csv = f"{CC_csv_dir}/average_grayscale_per_wsi.csv"


results = []


for wsi_id in os.listdir(CC_csv_dir):
    wsi_path = os.path.join(CC_csv_dir, wsi_id)
    if not os.path.isdir(wsi_path):
        continue

    wsi_id = int(wsi_id)
    csv_file = f"{wsi_path}/1{wsi_id:04d}_patch_in_region_filter_2_v2.csv"
    if not os.path.exists(csv_file):
        continue

    # 預先計算 CSV 總行數，確保 tqdm 進度條顯示完整數量
    total_rows = sum(1 for _ in open(csv_file)) - 1  # 減去標題行

    # 讀取 CSV（使用 iterator 以節省記憶體）
    df_iter = pd.read_csv(csv_file, chunksize=1000)

    # 初始化均值變數
    N_avg_gray, C_avg_gray = 0, 0
    N_count, C_count = 0, 0  # 計算樣本數

    # 逐步處理 CSV
    with tqdm(total=total_rows, desc=f"WSI {wsi_id}", unit="rows") as pbar:
        for df in df_iter:
            for _, row in df.iterrows():
                patch_filename = row.iloc[0]  # 第一欄是檔名
                label = row.iloc[1]           # 第二欄是類別標籤

                if label not in ["N", "C"]:
                    continue  # 只計算 N 和 C 類別

                patch_path = f"{CC_patches_save_path}/{wsi_id}/{patch_filename}"
                if not os.path.exists(patch_path):
                    continue

                # 讀取影像並轉換為灰階
                image = cv2.imread(patch_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue

                avg_gray = np.mean(image)  # 計算平均灰階

                # 增量更新
                if label == "N":
                    N_count += 1
                    N_avg_gray += (avg_gray - N_avg_gray) / N_count
                elif label == "C":
                    C_count += 1
                    C_avg_gray += (avg_gray - C_avg_gray) / C_count
                
                pbar.update(1)

    # 直接寫入 CSV
    with open(output_csv, "a") as f:
        f.write(f"{wsi_id},{N_avg_gray},{C_avg_gray}\n")

print(f"result save in {output_csv}")
