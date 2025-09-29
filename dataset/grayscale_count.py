import os
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
import re

def count_labeled_avg():
    CC_csv_dir = "/workspace/Data/Results/CC_NDPI/Data_Info"
    CC_patches_save_path = "/workspace/Data/Datas/CC_Patch"
    output_csv = f"{CC_csv_dir}/average_grayscale_per_wsi_new.csv"

    results = []

    for wsi_id in os.listdir(CC_csv_dir):
    # for wsi_id in ['131', '138', '165', '180', '184', '201', '202', '204']:
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
        # N_avg_gray, C_avg_gray = 0, 0
        # N_count, C_count = 0, 0  # 計算樣本數
        # 初始化均值變數 (灰階 + 三通道)
        stats = {
            "N": {"gray": 0, "b": 0, "g": 0, "r": 0, "count": 0},
            "C": {"gray": 0, "b": 0, "g": 0, "r": 0, "count": 0},
        }

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

                    # 讀取影像 (原彩色)
                    img = cv2.imread(patch_path)
                    if img is None:
                        pbar.update(1)
                        continue

                    # 灰階平均
                    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    avg_gray = np.mean(gray_image)

                    # 增量更新
                    # if label == "N":
                    #     N_count += 1
                    #     N_avg_gray += (avg_gray - N_avg_gray) / N_count
                    # elif label == "C":
                    #     C_count += 1
                    #     C_avg_gray += (avg_gray - C_avg_gray) / C_count

                    # 三通道平均 (cv2 是 BGR)
                    b_mean = np.mean(img[:, :, 0])
                    g_mean = np.mean(img[:, :, 1])
                    r_mean = np.mean(img[:, :, 2])

                    # 增量更新 (避免存全部數據)
                    s = stats[label]
                    s["count"] += 1
                    n = s["count"]
                    s["gray"] += (avg_gray - s["gray"]) / n
                    s["b"]    += (b_mean - s["b"]) / n
                    s["g"]    += (g_mean - s["g"]) / n
                    s["r"]    += (r_mean - s["r"]) / n

                    pbar.update(1)

        # 直接寫入 CSV
        # with open(output_csv, "a") as f:
        #     f.write(f"{wsi_id},{N_avg_gray},{C_avg_gray}\n")
        # 保存結果 (每個 WSI 一次)
        results.append({
            "wsi_id": wsi_id,
            "N_gray": stats["N"]["gray"], "N_b": stats["N"]["b"], "N_g": stats["N"]["g"], "N_r": stats["N"]["r"],
            "C_gray": stats["C"]["gray"], "C_b": stats["C"]["b"], "C_g": stats["C"]["g"], "C_r": stats["C"]["r"],
            "N_count": stats["N"]["count"], "C_count": stats["C"]["count"],
        })
        print("wsi_id: ", wsi_id,\
            "N_gray: ", stats["N"]["gray"], "N_b: ", stats["N"]["b"], "N_g: ", stats["N"]["g"], "N_r: ", stats["N"]["r"],\
            "C_gray: ", stats["C"]["gray"], "C_b: ", stats["C"]["b"], "C_g: ", stats["C"]["g"], "C_r: ", stats["C"]["r"],\
            "N_count: ", stats["N"]["count"], "C_count:", stats["C"]["count"])

    # 轉成 DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)

    # print(f"result save in {output_csv}")

def count_background_avg():
    CC_patches_save_path = "/workspace/Data/Datas/CC_Patch/131"

    grays = []
    b_vals, g_vals, r_vals = [], [], []

    for x in tqdm(range(0, 4480, 448)):
        pattern = re.compile(rf"^{x}_.+\.tif$")
        files = [f for f in os.listdir(CC_patches_save_path) if pattern.match(f)]

        for f in files:
            img_path = f"{CC_patches_save_path}/{f}"
            img = cv2.imread(img_path)
            
            color = ('b','g','r')
            bgr_cal = []
            for i, col in enumerate(color):
                histr = cv2.calcHist([img],[i],None,[256],[0, 256])
                bgr_cal.append(np.argmax(histr))
            b, g, r = bgr_cal
            if int(b == g == r == 0) == 1:
                continue

            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            grays.append(np.mean(gray_image))

            # channel means
            b_vals.append(np.mean(img[:, :, 0]))  # Blue channel
            g_vals.append(np.mean(img[:, :, 1]))  # Green channel
            r_vals.append(np.mean(img[:, :, 2]))  # Red channel
    
    print("Average grayscale:", np.mean(grays))
    print("Average Blue channel:", np.mean(b_vals))
    print("Average Green channel:", np.mean(g_vals))
    print("Average Red channel:", np.mean(r_vals))

count_labeled_avg()
# count_background_avg()