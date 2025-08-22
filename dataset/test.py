# import numpy as np
# import torch
# import pandas as pd
# import matplotlib.pyplot as plt

# print("debug")
# print(torch.cuda.is_available())

# train_loss_list = [0.194108502,0.12833176,0.109597589,0.09775072,0.085420897,0.077891686,0.069493554,0.062969841,0.05655]
# valid_loss_list = [0.454657939,0.173584941,0.113100196,0.091761335,0.088030437,0.108318072,0.065371412,0.806971064,0.44929]
# train_acc_list = [0.928885771,0.954655011,0.961660911,0.965761925,0.970105763,0.972920714,0.975609756,0.977849126,0.98038]
# valid_acc_list = [0.807609321,0.935270426,0.960802647,0.969073648,0.970943613,0.964470656,0.977056962,0.735327963,0.84717]

# # 轉成 DataFrame
# df = pd.DataFrame({
#     "train_loss": train_loss_list,
#     "valid_loss": valid_loss_list,
#     "train_acc": train_acc_list,
#     "valid_acc": valid_acc_list
# })

# # 定義輸出檔案路徑
# csv_path = "/workspace/graduate_project/dataset/epoch_log.csv"

# # 存成 CSV，不寫入 index
# # df.to_csv(csv_path, index=False)

# # plt.figure(figsize=(8, 6))
# epochs = range(1, len(train_loss_list) + 1)

# # plt.plot(epochs, train_loss_list, label="Train Loss", marker="o", linestyle="-", color="blue")
# # plt.plot(epochs, valid_loss_list, label="Valid Loss", marker="s", linestyle="-", color="red")

# # # 圖表設定
# # plt.xlabel("Epoch")
# # plt.ylabel("Loss")
# # plt.title("Train vs Valid Loss")
# # plt.legend()
# # plt.grid(True)

# # 創建圖表
# fig, ax1 = plt.subplots(figsize=(8, 6))

# # 繪製 loss（左側 Y 軸）
# l1 = ax1.plot(epochs, train_loss_list, label="Train Loss", marker="o", linestyle="-", color="blue")
# l2 = ax1.plot(epochs, valid_loss_list, label="Valid Loss", marker="s", linestyle="--", color="red")
# ax1.set_xlabel("Epoch")
# ax1.set_ylabel("Loss")
# ax1.tick_params(axis="y", labelcolor="blue")
# # ax1.legend(loc="upper left")
# ax1.grid(True, linestyle="--", alpha=0.5)

# # 創建第二個 Y 軸（右側）來顯示 accuracy
# ax2 = ax1.twinx()
# l3 = ax2.plot(epochs, train_acc_list, label="Train Accuracy", marker="^", linestyle="-", color="green")
# l4 = ax2.plot(epochs, valid_acc_list, label="Valid Accuracy", marker="D", linestyle="--", color="orange")
# ax2.set_ylabel("Accuracy")
# ax2.tick_params(axis="y", labelcolor="green")
# # ax2.legend(loc="lower right")

# lines = l1 + l2 + l3 + l4
# labels = [l.get_label() for l in lines]
# ax1.legend(lines, labels, loc="upper left")

# # 設定標題
# plt.title("Train vs Valid Loss & Accuracy")


# # 顯示圖表
# plt.savefig("loss_and_accuracy_curve.png", dpi=300, bbox_inches="tight")


# # train_loss_arr = np.array(train_loss_list).reshape(1,len(train_loss_list))
# # valid_loss_arr = np.array(valid_loss_list).reshape(1,len(train_loss_list))
# # train_acc_arr = np.array(train_acc_list).reshape(1,len(train_loss_list))
# # valid_acc_arr = np.array(valid_acc_list).reshape(1,len(train_loss_list))

# # training_log = np.append(train_loss_arr,valid_loss_arr,axis=0)
# # training_log = np.append(training_log,train_acc_arr,axis=0)
# # training_log = np.append(training_log,valid_acc_arr,axis=0)

# # np.savetxt(f"{loss_save_path}/{condition}_epoch_log.csv", header = "train_loss,valid_loss,train_acc,valid_acc", delimiter=",",comments="")

# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np

# # 讀取 CSV 檔案
# base_folder = "/workspace/Data/Results/CC_NDPI/Data_Info"
# file_path = f"{base_folder}/average_grayscale_per_wsi_new.csv"
# data = pd.read_csv(file_path)

# # 設定 X 軸範圍（整數刻度）
# bin_edges = np.arange(int(data[['N_avg_gray', 'C_avg_gray']].min().min()), 
#                       int(data[['N_avg_gray', 'C_avg_gray']].max().max()) + 2, 1)

# # 計算每個區間的數量
# n_counts, _ = np.histogram(data['N_avg_gray'], bins=bin_edges)
# c_counts, _ = np.histogram(data['C_avg_gray'], bins=bin_edges)

# # 設定 X 軸標籤（對應 bin 的中心值）
# x_labels = bin_edges[:-1] + 0.5  # 讓長條圖對齊區間

# # 繪製 N_avg_gray 圖
# plt.figure(figsize=(20, 12))
# plt.bar(x_labels, n_counts, width=0.8, color='blue', alpha=0.7)
# plt.xticks(np.arange(bin_edges.min(), bin_edges.max(), 1))  # X 軸整數間隔 1
# plt.xlabel("Gray Value")
# plt.ylabel("Data Count")
# plt.title("Histogram of N_avg_gray", fontsize=32)
# plt.grid(axis='y', linestyle='--', alpha=0.6)

# # 儲存圖片
# plt.savefig(f"{base_folder}/N_avg_gray_histogram.png", dpi=300)
# plt.close()  # 關閉當前圖表，避免影響下一張圖

# # 繪製 C_avg_gray 圖
# plt.figure(figsize=(20, 12))
# plt.bar(x_labels, c_counts, width=0.8, color='orange', alpha=0.7)
# plt.xticks(np.arange(bin_edges.min(), bin_edges.max(), 1))  # X 軸整數間隔 1
# plt.xlabel("Gray Value")
# plt.ylabel("Data Count")
# plt.title("Histogram of C_avg_gray", fontsize=32)
# plt.grid(axis='y', linestyle='--', alpha=0.6)

# # 儲存圖片
# plt.savefig(f"{base_folder}/C_avg_gray_histogram.png", dpi=300)
# plt.close()  # 關閉當前圖表

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# from sklearn.metrics import confusion_matrix

# all_labels = []
# all_preds = []

# save_dir = "/workspace/Data/Results/Mix_NDPI/10WTC_Result/LP_6400/trial_2"

# wsis =  [6, 11, 39, 52, 144]

# label_dict = {"N": 0, "H": 1, "C": 2}

# for _wsi in wsis:
#     # wsi = _wsi
#     wsi = f"1{_wsi:04d}"
#     save_path = f"{save_dir}/{wsi}"
#     condition = f"{wsi}_10WTC_LP6400_3_class_trial_2" 
#     df = pd.read_csv(f"{save_path}/Metric/{condition}_labels_predictions.csv")
#     labels = df['true_label'].to_list()
#     preds = df['pred_label'].to_list()
#     for label, pred in zip(labels, preds):
#         all_labels.append(label_dict[label])
#         all_preds.append(label_dict[pred])

# # 把你的原資料複製一份
# all_labels_fixed = np.array(all_labels).tolist()
# all_preds_fixed = np.array(all_preds).tolist()

# # 人為加上每一個 label 的假樣本（預設 prediction 也設成自己）
# for label in [0, 1, 2]:
#     all_labels_fixed.append(label)
#     all_preds_fixed.append(label)

# # 計算 confusion matrix
# cm = confusion_matrix(all_labels_fixed, all_preds_fixed, labels=[0, 1, 2])

# # 最後把加的那個 fake 样本在 cm 中扣掉 (每個 fake 樣本只增加 1 到對角線)
# cm = cm - np.eye(3, dtype=int)

# condition = "CC_tani_trial_2"
# # cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])

# fig, ax = plt.subplots(figsize=(8, 6))
# cax = ax.matshow(cm, cmap='Blues')
# fig.colorbar(cax)

# ax.set_xticks(np.arange(3))
# ax.set_yticks(np.arange(3))
# ax.set_xticklabels(["N", "H", "C"], fontsize=14)
# ax.set_yticklabels(["N", "H", "C"], fontsize=14)

# for i in range(cm.shape[0]):
#     for j in range(cm.shape[1]):
#         color = "white" if cm[i, j] > cm.max() / 2 else "black"
#         ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color=color, fontsize=18)

# title = f"Confusion Matrix of {condition}"
# plt.title(title, fontsize=20, pad=20)
# ax.set_xlabel('Predicted Label', fontsize=16, labelpad=10)
# ax.set_ylabel('True Label', fontsize=16, labelpad=10)

# plt.subplots_adjust(top=0.85)
# plt.savefig(f"{save_dir}/Metric/{condition}_confusion_matrix.png")


import os
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

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