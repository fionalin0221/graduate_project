import numpy as np
import torch
import os
import cv2
import shutil
import re
import glob
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score, confusion_matrix

# print(torch.cuda.is_available())

def loss_curve():
    train_loss_list = [0.194108502,0.12833176,0.109597589,0.09775072,0.085420897,0.077891686,0.069493554,0.062969841,0.05655]
    valid_loss_list = [0.454657939,0.173584941,0.113100196,0.091761335,0.088030437,0.108318072,0.065371412,0.806971064,0.44929]
    train_acc_list = [0.928885771,0.954655011,0.961660911,0.965761925,0.970105763,0.972920714,0.975609756,0.977849126,0.98038]
    valid_acc_list = [0.807609321,0.935270426,0.960802647,0.969073648,0.970943613,0.964470656,0.977056962,0.735327963,0.84717]

    # 轉成 DataFrame
    df = pd.DataFrame({
        "train_loss": train_loss_list,
        "valid_loss": valid_loss_list,
        "train_acc": train_acc_list,
        "valid_acc": valid_acc_list
    })

    # 定義輸出檔案路徑
    csv_path = "/workspace/graduate_project/dataset/epoch_log.csv"

    # 存成 CSV，不寫入 index
    # df.to_csv(csv_path, index=False)

    # plt.figure(figsize=(8, 6))
    epochs = range(1, len(train_loss_list) + 1)

    # plt.plot(epochs, train_loss_list, label="Train Loss", marker="o", linestyle="-", color="blue")
    # plt.plot(epochs, valid_loss_list, label="Valid Loss", marker="s", linestyle="-", color="red")

    # # 圖表設定
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.title("Train vs Valid Loss")
    # plt.legend()
    # plt.grid(True)

    # 創建圖表
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # 繪製 loss（左側 Y 軸）
    l1 = ax1.plot(epochs, train_loss_list, label="Train Loss", marker="o", linestyle="-", color="blue")
    l2 = ax1.plot(epochs, valid_loss_list, label="Valid Loss", marker="s", linestyle="--", color="red")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.tick_params(axis="y", labelcolor="blue")
    # ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 創建第二個 Y 軸（右側）來顯示 accuracy
    ax2 = ax1.twinx()
    l3 = ax2.plot(epochs, train_acc_list, label="Train Accuracy", marker="^", linestyle="-", color="green")
    l4 = ax2.plot(epochs, valid_acc_list, label="Valid Accuracy", marker="D", linestyle="--", color="orange")
    ax2.set_ylabel("Accuracy")
    ax2.tick_params(axis="y", labelcolor="green")
    # ax2.legend(loc="lower right")

    lines = l1 + l2 + l3 + l4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")

    # 設定標題
    plt.title("Train vs Valid Loss & Accuracy")


    # 顯示圖表
    plt.savefig("loss_and_accuracy_curve.png", dpi=300, bbox_inches="tight")


    train_loss_arr = np.array(train_loss_list).reshape(1,len(train_loss_list))
    valid_loss_arr = np.array(valid_loss_list).reshape(1,len(train_loss_list))
    train_acc_arr = np.array(train_acc_list).reshape(1,len(train_loss_list))
    valid_acc_arr = np.array(valid_acc_list).reshape(1,len(train_loss_list))

    training_log = np.append(train_loss_arr,valid_loss_arr,axis=0)
    training_log = np.append(training_log,train_acc_arr,axis=0)
    training_log = np.append(training_log,valid_acc_arr,axis=0)

    np.savetxt(f"{loss_save_path}/{condition}_epoch_log.csv", header = "train_loss,valid_loss,train_acc,valid_acc", delimiter=",",comments="")

def count_labeled_avg():
    CC_csv_dir = "/workspace/Data/Results/CC_NDPI/Data_Info"
    CC_patches_save_path = "/workspace/Data/Datas/CC_Patch"
    output_csv = f"{CC_csv_dir}/average_grayscale_per_wsi_new.csv"

    results = []

    for wsi_id in os.listdir(CC_csv_dir):
        wsi_path = os.path.join(CC_csv_dir, wsi_id)
        if not os.path.isdir(wsi_path):
            continue

        wsi_id = int(wsi_id)
        csv_file = f"{wsi_path}/1{wsi_id:04d}_patch_in_region_filter_2_v2.csv"
        if not os.path.exists(csv_file):
            continue

        total_rows = sum(1 for _ in open(csv_file)) - 1
        df_iter = pd.read_csv(csv_file, chunksize=1000)

        stats = {
            "N": {"gray": 0, "b": 0, "g": 0, "r": 0, "count": 0},
            "C": {"gray": 0, "b": 0, "g": 0, "r": 0, "count": 0},
        }

        with tqdm(total=total_rows, desc=f"WSI {wsi_id}", unit="rows") as pbar:
            for df in df_iter:
                for _, row in df.iterrows():
                    patch_filename = row.iloc[0]
                    label = row.iloc[1]

                    if label not in ["N", "C"]:
                        continue

                    patch_path = f"{CC_patches_save_path}/{wsi_id}/{patch_filename}"
                    if not os.path.exists(patch_path):
                        continue

                    img = cv2.imread(patch_path)
                    if img is None:
                        pbar.update(1)
                        continue

                    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    avg_gray = np.mean(gray_image)

                    b_mean = np.mean(img[:, :, 0])
                    g_mean = np.mean(img[:, :, 1])
                    r_mean = np.mean(img[:, :, 2])

                    s = stats[label]
                    s["count"] += 1
                    n = s["count"]
                    s["gray"] += (avg_gray - s["gray"]) / n
                    s["b"]    += (b_mean - s["b"]) / n
                    s["g"]    += (g_mean - s["g"]) / n
                    s["r"]    += (r_mean - s["r"]) / n

                    pbar.update(1)

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

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)

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

def plot_gray_histogram():
    # 讀取 CSV 檔案
    base_folder = "/workspace/Data/Results/CC_NDPI/Data_Info"
    file_path = f"{base_folder}/average_grayscale_per_wsi_new.csv"
    data = pd.read_csv(file_path)

    # 設定 X 軸範圍（整數刻度）
    bin_edges = np.arange(int(data[['N_avg_gray', 'C_avg_gray']].min().min()), 
                        int(data[['N_avg_gray', 'C_avg_gray']].max().max()) + 2, 1)

    # 計算每個區間的數量
    n_counts, _ = np.histogram(data['N_avg_gray'], bins=bin_edges)
    c_counts, _ = np.histogram(data['C_avg_gray'], bins=bin_edges)

    # 設定 X 軸標籤（對應 bin 的中心值）
    x_labels = bin_edges[:-1] + 0.5  # 讓長條圖對齊區間

    # 繪製 N_avg_gray 圖
    plt.figure(figsize=(20, 12))
    plt.bar(x_labels, n_counts, width=0.8, color='blue', alpha=0.7)
    plt.xticks(np.arange(bin_edges.min(), bin_edges.max(), 1))  # X 軸整數間隔 1
    plt.xlabel("Gray Value")
    plt.ylabel("Data Count")
    plt.title("Histogram of N_avg_gray", fontsize=32)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # 儲存圖片
    plt.savefig(f"{base_folder}/N_avg_gray_histogram.png", dpi=300)
    plt.close()  # 關閉當前圖表，避免影響下一張圖

    # 繪製 C_avg_gray 圖
    plt.figure(figsize=(20, 12))
    plt.bar(x_labels, c_counts, width=0.8, color='orange', alpha=0.7)
    plt.xticks(np.arange(bin_edges.min(), bin_edges.max(), 1))  # X 軸整數間隔 1
    plt.xlabel("Gray Value")
    plt.ylabel("Data Count")
    plt.title("Histogram of C_avg_gray", fontsize=32)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # 儲存圖片
    plt.savefig(f"{base_folder}/C_avg_gray_histogram.png", dpi=300)
    plt.close()  # 關閉當前圖表

def threshold_classification():
    classes = ["N", "H", "C"]
    invalid_idx = len(classes)
    classes_with_invalid = classes + ["Invalid"]

    base_path = "/home/ipmclab-2/project/Results/Mix_NDPI/100WTC_Result/LP_3200/trial_1"
    wsis = sorted(os.listdir(base_path))

    for wsi in wsis:
        if wsi in ["Data", "Loss", "Metric", "Model", "TI"]:
            continue
        
        print(wsi)
        condition = f"{wsi}_100WTC_LP3200_3_class_trial_1"

        # Load the saved labels/predictions
        df_results = pd.read_csv(f"{base_path}/Metric/{condition}_labels_predictions.csv")

        # Map labels back to indices
        all_labels = [classes.index(l) for l in df_results["true_label"]]
        all_preds = []
        for pred in df_results["pred_label"]:
            if pd.isna(pred) or pred == "None":
                all_preds.append(invalid_idx)
            elif pred in classes:
                all_preds.append(classes.index(pred))
            else:
                # case where pred_label is "2", "3", etc. (ambiguous multi-class)
                all_preds.append(invalid_idx)

        # Compute metrics
        acc = accuracy_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds, labels=range(len(classes_with_invalid)))

        print(cm)
        print("Accuracy: {:.4f}".format(acc))

        # Collect per-class stats
        Test_Acc = {"Condition": [condition], "Accuracy": [acc]}
        for i, class_name in enumerate(classes):
            TP = cm[i, i]
            FN = cm[i, :].sum() - TP
            FP = cm[:, i].sum() - TP
            TN = cm.sum() - (TP + FP + FN)

            Test_Acc[f"{class_name}_TP"] = [TP]
            Test_Acc[f"{class_name}_FN"] = [FN]
            Test_Acc[f"{class_name}_TN"] = [TN]
            Test_Acc[f"{class_name}_FP"] = [FP]

        # Save new test results
        pd.DataFrame(Test_Acc).to_csv(f"{base_path}/Metric/{condition}_test_result.csv", index=False)

def find_csv_duplicate():
    df = pd.read_csv("/workspace/Data/Results/Mix_NDPI/Generation_Training/100WTC_LP_3200/10138/trial_1/Data/10138_Gen1_ND_zscore_selected_patches_by_Gen0.csv")
    duplicates = df[df.duplicated(subset=["file_name"], keep=False)]
    print(duplicates)

def diff_of_two_image():
    img1 = cv2.imread("contour_filtered_result_copy.png", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("contour_filtered_result.png", cv2.IMREAD_GRAYSCALE)

    diff_abs = cv2.absdiff(img1, img2)
    cv2.imwrite("diff_abs.png", diff_abs)

def threshold_classification_analysis():
    classes = ["N", "H", "C"]
    invalid_idx = len(classes)
    classes_with_invalid = classes + ["Invalid"]

    # Define the 5 invalid cases
    invalid_cases = {
        (0, 0, 0): "000",
        (1, 1, 0): "110",
        (1, 0, 1): "101",
        (0, 1, 1): "011",
        (1, 1, 1): "111",
    }
    # Create empty storage for all cases across all WSIs
    case_data = {case: [] for case in invalid_cases.values()}

    base_path = "/home/ipmclab-2/project/Results/Mix_NDPI/100WTC_Result/LP_3200/trial_1"

    wsis = sorted(os.listdir(base_path))

    for wsi in tqdm(wsis):
        # if wsi in ["Data", "Loss", "Metric", "Model", "TI"]:
        #     continue
        if not wsi.isdigit():  # skip if not only numbers
            continue
        
        condition = f"{wsi}_100WTC_LP3200_3_class_trial_1"

        # Load the saved labels/predictions
        df = pd.read_csv(f"{base_path}/TI/{wsi}_100WTC_LP3200_3_class_trial_1_patch_in_region_filter_2_v2_TI_with_threshold.csv")
        df_labels = pd.read_csv(f"{base_path}/Metric/{condition}_labels_predictions.csv")
        df_merged = df.merge(df_labels[["file_name", "true_label"]], on="file_name", how="left")

        for _, row in df_merged.iterrows():
            if row["sum_bin"] != 1:
                case_key = (row["N_pred_bin"], row["H_pred_bin"], row["C_pred_bin"])
                if case_key in invalid_cases:
                    case_name = invalid_cases[case_key]

                    # compute predicted label by argmax of [N_pred, H_pred, C_pred]
                    preds = [row["N_pred"], row["H_pred"], row["C_pred"]]
                    pred_label = classes[int(pd.Series(preds).idxmax())]

                    case_data[case_name].append([
                        wsi,
                        row["file_name"],
                        row["N_pred"],
                        row["H_pred"],
                        row["C_pred"],
                        pred_label,
                        row["true_label"]
                    ])


    for case_name, rows in case_data.items():
        if rows:  # Only save if non-empty
            df_out = pd.DataFrame(rows, columns=["WSI", "file_name", "N_pred", "H_pred", "C_pred", "pred_label","true_label"])
            df_out.to_csv(f"{base_path}/invalid_case_{case_name}.csv", index=False)

def rename():
    base_dir = "/home/ipmclab/project/Results/CC_NDPI/40WTC_Result/LP_3200/trial_4"  # change to your root directory
    old_str = "_trial_1_"
    new_str = "_trial_4_"

    for root, dirs, files in os.walk(base_dir):
        for filename in files:
            if old_str in filename:
                old_path = os.path.join(root, filename)
                new_filename = filename.replace(old_str, new_str)
                new_path = os.path.join(root, new_filename)
                
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} → {new_path}")

def delete():
    # Specify the directory containing images
    image_dir = "/workspace/Data/Datas/CC_Patch"

    # Specify image extensions to delete (e.g., PNG, JPG)
    extensions = ["*.tif"]

    # Loop through each extension and delete matching files
    for ext in extensions:
        for image_path in glob.glob(os.path.join(image_dir, ext)):
            try:
                os.remove(image_path)
                print(f"Deleted: {image_path}")
            except Exception as e:
                print(f"Error deleting {image_path}: {e}")

    print("Image deletion completed.")

def result_collect_another_class():
    result_type = "CC"
    num_wsi = 100
    data_num = "ALL"
    num_trial = 1
    num_class = 2
    base_path = f"/home/ipmclab/project/Results/{result_type}_NDPI/{num_wsi}WTC_Result/LP_{data_num}"
    output_file = f"{base_path}/{num_wsi}WTC_LP{data_num}_trial_{num_trial}_hcc_test_results.csv"
    HCC_wsi_list = [105, 117, 133, 151, 153, 154, 159, 160, 168, 169, 170, 171, 178, 180, 181, 183, 186, 189, 190, 194]
    # HCC_wsi_list = []
    # CC_wsi_list =  [373, 376, 377, 378, 379, 380, 390, 391, 392, 400, 401, 402, 406, 407, 408, 409, 410, 422, 454, 455]
    CC_wsi_list = []

    def add_results(file_path, cl, wsi, num_trial, results):
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return results  # Skip if file does not exist

        # Read the CSV file
        df = pd.read_csv(file_path)
        if cl == 'H':
            true_is_cl = df["true_label"] == 'C'
            pred_is_cl = df["pred_label"] == 'C'
        else:
            true_is_cl = df["true_label"] == 'N'
            pred_is_cl = df["pred_label"] == 'N'
        TP = ((true_is_cl) & (pred_is_cl)).sum()
        FN = ((true_is_cl) & (~pred_is_cl)).sum()
        results.append([num_trial, wsi, cl, TP, FN])

        return results

    def collect_results(wsi, cl, results):
        file_path = f"{base_path}/trial_{num_trial}/{wsi}/Metric/{wsi}_{num_wsi}WTC_LP{data_num}_{num_class}_class_trial_{num_trial}_for_epoch_18_labels_predictions.csv"
            
        results = add_results(file_path, cl, wsi, num_trial, results)

        return results

    # Initialize result list
    results = []

    # Iterate through trials and WSIs
    for _wsi in HCC_wsi_list:
        wsi = _wsi
        results = collect_results(wsi, "H", results)

    for _wsi in CC_wsi_list:
        wsi = f"1{_wsi:04d}"
        results = collect_results(wsi, "C", results)

    for _wsi in HCC_wsi_list:
        wsi = _wsi
        results = collect_results(wsi, "N", results)

    for _wsi in CC_wsi_list:
        wsi = f"1{_wsi:04d}"
        results = collect_results(wsi, "N", results)
                
    # --- Convert to DataFrame ---
    df = pd.DataFrame(results, columns=['Trial', 'WSI', 'Class', 'TP', 'FN'])

    # --- Separate tumor (H/C) and normal (N) ---
    df_tumor = df[df['Class'].isin(['H', 'C'])].copy()
    df_normal = df[df['Class'] == 'N'].copy()

    # Rename N columns (TP→TN, FN→FP)
    df_normal = df_normal.rename(columns={'TP': 'TN', 'FN': 'FP'})

    # Merge on Trial, WSI, Gen, Condition
    merged = pd.merge(
        df_tumor,
        df_normal[['Trial', 'WSI', 'TN', 'FP']],
        on=['Trial', 'WSI'],
        how='left'
    )

    # --- Final columns (remove Class) ---
    df_output = merged[['Trial', 'WSI', 'TP', 'FN', 'TN', 'FP']]
    print(df_output)
    df_output.to_csv(output_file, index=False)

    print(f"Processed results saved to {output_file}")

def result_collect():
    result_type = "Mix"
    num_wsi = 100
    data_num = 3200
    num_trial = 1
    gen = 1
    num_class = 3

    HCC_wsi_list = [105, 117, 133, 151, 153, 154, 159, 160, 168, 169, 170, 171, 178, 180, 181, 183, 186, 189, 190, 194]
    CC_wsi_list =  [373, 376, 377, 378, 379, 380, 390, 391, 392, 400, 401, 402, 406, 407, 408, 409, 410, 422, 454, 455]

    # base_path = f"/workspace/Data/Results/{result_type}_NDPI/Generation_Training/{num_wsi}WTC_LP_{data_num}"
    # base_path = f"/home/ipmclab-2/project/Results/{result_type}_NDPI/Generation_Training/{num_wsi}WTC_LP_{data_num}"
    base_path = f"/home/ipmclab/project/Results/{result_type}_NDPI/Generation_Training/{num_wsi}WTC_LP_{data_num}"
    output_file = f"{base_path}/{num_wsi}WTC_LP{data_num}_trial_{num_trial}_generation_confusion_matrix.csv"

    def flatten_cm(wsi, gen, pred_type):
        if pred_type == 'inference':
            if gen == 0:
                file_path = f"{base_path}/{wsi}/trial_{num_trial}/Metric/{wsi}_{num_class}_class_confusion_matrix.csv" 
            else:
                file_path = f"{base_path}/{wsi}/trial_{num_trial}/Metric/{wsi}_Gen{gen}_ND_zscore_selected_patches_by_Gen{gen-1}_confusion_matrix.csv" 
        elif pred_type == 'flip':
            file_path = f"{base_path}/{wsi}/trial_{num_trial}/Metric/{wsi}_Gen{gen}_ND_zscore_selected_patches_by_Gen{gen-1}_flip_confusion_matrix.csv" 
        df_cm = pd.read_csv(file_path, index_col=0)

        cm_flattened = df_cm.stack()
        flattened_data = {}
        for (true_lab, pred_lab), value in cm_flattened.items():
            column_name = f"{true_lab}_{pred_lab}"
            flattened_data[column_name] = [value]

        df_flat = pd.DataFrame(flattened_data)
        df_flat.insert(0, 'WSI', wsi)
        df_flat.insert(1, 'Trial', num_trial)
        df_flat.insert(2, 'Generation', gen)
        df_flat.insert(3, 'Condition', pred_type)

        return df_flat

    all_results = []

    for wsi in HCC_wsi_list:
        result = flatten_cm(wsi, 0, 'inference')
        all_results.append(result)
        result = flatten_cm(wsi, 1, 'flip')
        all_results.append(result)
    for wsi in CC_wsi_list:
        wsi = f"1{wsi:04d}"
        result = flatten_cm(wsi, 0, 'inference')
        all_results.append(result)
        result = flatten_cm(wsi, 1, 'flip')
        all_results.append(result)

    final_summary_df = pd.concat(all_results, ignore_index=True)
    final_summary_df.to_csv(output_file, index=False)

        
def sample_patches():
    seed = 42
    valid_HCC_wsis = [14, 26, 42, 60, 62, 63, 68, 69, 77, 78, 79, 80, 87, 89, 90, 92, 95, 98, 99, 103]
    valid_CC_wsis = [373, 376, 377, 378, 379, 380, 390, 391, 392, 400, 401, 402, 406, 407, 408, 409, 410, 422, 454, 455]

    def move_patches(wsi_type, wsi, sampled_df):
        src_folder = f"/workspace/temp_folder/{wsi_type}_Patch/{wsi}"
        dst_folder = f"/workspace/Data/Datas/{wsi_type}_Patch/{wsi}"
        os.makedirs(dst_folder, exist_ok=True)
        for filename in tqdm(sampled_df['file_name']):
            src = os.path.join(src_folder, filename)
            dst = os.path.join(dst_folder, filename)
            shutil.copy2(src, dst)

    for wsi in valid_HCC_wsis:
        base_path = "/workspace/Data/Results/HCC_NDPI/Data_Info"
        df = pd.read_csv(f"{base_path}/{wsi+91}/{wsi+91}_patch_in_region_filter_2_v2.csv")
        sampled_df = pd.concat([
            df[df['label'] == label].sample(n=100, random_state=seed) 
            for label in ['N', 'H']
        ])
        sampled_df.to_csv(f"{base_path}/{wsi+91}/{wsi+91}_patch_in_region_filter_2_v2_sampled.csv", index=False)
        move_patches("HCC", wsi, sampled_df)

    for wsi in valid_CC_wsis:
        base_path = "/workspace/Data/Results/CC_NDPI/Data_Info"
        df = pd.read_csv(f"{base_path}/{wsi}/1{wsi:04d}_patch_in_region_filter_2_v2.csv")
        sampled_df = pd.concat([
            df[df['label'] == label].sample(n=100, random_state=seed) 
            for label in ['N', 'C']
        ])
        sampled_df.to_csv(f"{base_path}/{wsi}/1{wsi:04d}_patch_in_region_filter_2_v2_sampled.csv", index=False)
        move_patches("CC", wsi, sampled_df)

sample_patches()