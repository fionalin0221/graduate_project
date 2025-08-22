import pandas as pd
import os
import numpy as np
from tqdm import tqdm
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
    if wsi in ["1", "2", "3", "4", "5"]:
        csv_path = f"{base_path}/{wsi}/TI/{wsi}_100WTC_LP3200_3_class_trial_1_patch_in_region_filter_2_v2_TI_new.csv"
    else:
        csv_path = f"{base_path}/{wsi}/TI/{wsi}_100WTC_LP3200_3_class_trial_1_patch_in_region_filter_2_v2_TI.csv"
    if not os.path.exists(csv_path):
        print(f"CSV not found for {wsi}, skipping.")
        continue

    df = pd.read_csv(csv_path)
    # Threshold each prediction at 0.5 → binary 0/1
    for col in ["N_pred", "H_pred", "C_pred"]:
        df[col + "_bin"] = (df[col] > 0.5).astype(int)

    # Sum the binary results across the three preds
    df["sum_bin"] = df[["N_pred_bin", "H_pred_bin", "C_pred_bin"]].sum(axis=1)
    # for _, row in df.iterrows():
    #     if row["sum_bin"] != 1:
    #         print(wsi, row["file_name"], row["sum_bin"])

    # Save result to new CSV
    df.to_csv(f"{base_path}/TI/{wsi}_100WTC_LP3200_3_class_trial_1_patch_in_region_filter_2_v2_TI_with_threshold.csv", index=False)

    if len(wsi) == 5:
        _wsi = int(wsi[1:]) 
        data_info_df = pd.read_csv(f'/home/ipmclab-2/project/Results/CC_NDPI/Data_Info/{_wsi}/{wsi}_patch_in_region_filter_2_v2.csv')
    else:
        data_info_df = pd.read_csv(f'/home/ipmclab-2/project/Results/HCC_NDPI/Data_Info/{wsi}/{wsi}_patch_in_region_filter_2_v2.csv')
    

    condition = f"{wsi}_100WTC_LP3200_3_class_trial_1"

    results_df = {"file_name":[], "true_label":[], "pred_label": []}
    all_labels, all_preds = [], []

    match_df  = data_info_df[data_info_df['file_name'].isin(df['file_name'])]
    filename_inRegion = match_df['file_name'].to_list()
    label_inRegion = match_df['label'].to_list()

    for idx, filename in enumerate(tqdm(filename_inRegion)):
        gt_label = label_inRegion[idx]
        gt_idx = classes.index(gt_label)

        results_df["file_name"].append(filename)
        row = df[df['file_name'] == filename].iloc[0]

        sum_bin = row["sum_bin"]

        if sum_bin == 0:
            pred_label = None   # no class chosen
            pred_idx = invalid_idx
        elif sum_bin == 1:
            # find which class has bin==1
            for i, cl in enumerate(classes):
                if row[f"{cl}_pred_bin"] == 1:
                    pred_label = cl
                    pred_idx = i
                    break
        else:  # sum_bin > 1
            pred_label = str(int(sum_bin))  # mark as "2" or "3"
            pred_idx = invalid_idx

        results_df["true_label"].append(gt_label)
        results_df["pred_label"].append(pred_label)

        all_labels.append(gt_idx)
        all_preds.append(pred_idx)

    # Save to CSV
    pd.DataFrame(results_df).to_csv(f"{base_path}/Metric/{condition}_labels_predictions.csv", index=False)

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(classes_with_invalid)))
    print("Accuracy: {:.4f}".format(acc))

    Test_Acc = {"Condition": [condition], "Accuracy": [acc]}
    for i, class_name in enumerate(classes):
        TP = cm[i, i]  # True Positives
        FN = cm[i, :].sum() - TP  # False Negatives
        FP = cm[:, i].sum() - TP  # False Positives
        TN = cm.sum() - (TP + FP + FN)  # True Negatives
        
        Test_Acc[f"{class_name}_TP"] = [TP]
        Test_Acc[f"{class_name}_FN"] = [FN]
        Test_Acc[f"{class_name}_TN"] = [TN]
        Test_Acc[f"{class_name}_FP"] = [FP]

    # Save to CSV
    pd.DataFrame(Test_Acc).to_csv(f"{base_path}/Metric/{condition}_test_result.csv", index=False)
