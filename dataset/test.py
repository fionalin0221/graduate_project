import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import cv2
from tqdm import tqdm

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

def plot_confusion_matrix():
    all_labels = []
    all_preds = []

    save_dir = "/workspace/Data/Results/Mix_NDPI/10WTC_Result/LP_6400/trial_2"

    wsis =  [6, 11, 39, 52, 144]

    label_dict = {"N": 0, "H": 1, "C": 2}

    for _wsi in wsis:
        # wsi = _wsi
        wsi = f"1{_wsi:04d}"
        save_path = f"{save_dir}/{wsi}"
        condition = f"{wsi}_10WTC_LP6400_3_class_trial_2" 
        df = pd.read_csv(f"{save_path}/Metric/{condition}_labels_predictions.csv")
        labels = df['true_label'].to_list()
        preds = df['pred_label'].to_list()
        for label, pred in zip(labels, preds):
            all_labels.append(label_dict[label])
            all_preds.append(label_dict[pred])

    # 把你的原資料複製一份
    all_labels_fixed = np.array(all_labels).tolist()
    all_preds_fixed = np.array(all_preds).tolist()

    # 人為加上每一個 label 的假樣本（預設 prediction 也設成自己）
    for label in [0, 1, 2]:
        all_labels_fixed.append(label)
        all_preds_fixed.append(label)

    # 計算 confusion matrix
    cm = confusion_matrix(all_labels_fixed, all_preds_fixed, labels=[0, 1, 2])

    # 最後把加的那個 fake 样本在 cm 中扣掉 (每個 fake 樣本只增加 1 到對角線)
    cm = cm - np.eye(3, dtype=int)

    condition = "CC_tani_trial_2"
    # cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(cm, cmap='Blues')
    fig.colorbar(cax)

    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(["N", "H", "C"], fontsize=14)
    ax.set_yticklabels(["N", "H", "C"], fontsize=14)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color=color, fontsize=18)

    title = f"Confusion Matrix of {condition}"
    plt.title(title, fontsize=20, pad=20)
    ax.set_xlabel('Predicted Label', fontsize=16, labelpad=10)
    ax.set_ylabel('True Label', fontsize=16, labelpad=10)

    plt.subplots_adjust(top=0.85)
    plt.savefig(f"{save_dir}/Metric/{condition}_confusion_matrix.png")

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
    # 找出 file_name 欄位中重複的列
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
    base_dir = "/home/ipmclab/project/Results/CC_NDPI/40WTC_Result/LP_3200/trial_4"  # <-- change to your root directory
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