import os
import pandas as pd

# Define paths
result_type = "Mix"
num_wsi = 20
data_num = "ALL"
num_trial = 6
num_class = 4
ep = 0

# base_path = f"/workspace/Data/Results/{result_type}_NDPI/{num_wsi}WTC_Result/LP_{data_num}"
# base_path = f"/home/ipmclab-2/project/Results/{result_type}_NDPI/{num_wsi}WTC_Result/LP_{data_num}"
base_path = f"/home/ipmclab/project/Results/{result_type}_NDPI/{num_wsi}WTC_Result/LP_{data_num}"
output_file = f"{base_path}/{num_wsi}WTC_LP{data_num}_trial_{num_trial}_test_results.csv"
# if ep == 0:
#     output_file = f"{base_path}/{num_wsi}WTC_LP{data_num}_trial_{num_trial}_test_results.csv"
# else:
#     output_file = f"{base_path}/{num_wsi}WTC_LP{data_num}_trial_{num_trial}_for_epoch_{ep}_tati_test_results.csv"

# Define trials and WSIs
# HCC_wsi_list = []
# CC_wsi_list = []

# HCC 10WTC
# HCC_wsi_list = [1, 6, 8, 13, 16, 27, 29, 30, 44, 45]
# HCC_wsi_list = [1, 12, 22, 33, 45, 56, 67, 76, 89, 91]
# HCC_wsi_list = [6, 11, 39, 52, 144]
# HCC_wsi_list = [1, 6, 8, 13, 16, 27, 29, 30, 31, 36, 38, 39, 44, 45, 47, 48, 49, 50, 51, 53]
HCC_wsi_list = [1, 6, 8, 13, 16, 31, 36, 38, 39, 44, 47, 48, 49, 50, 51, 53, 9, 19, 24, 25]

# CC 10WTC
# CC_wsi_list = [72, 108, 111, 116, 122, 124, 130, 131, 137, 138]
# CC_wsi_list = [2, 21, 50, 69, 81]
CC_wsi_list = [1, 2, 3, 6, 7, 8, 11, 12, 13, 14, 15, 39, 52, 53, 54, 55, 67, 69, 70, 71]

# HCC 40WTC
# HCC_wsi_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 37, 38, 39, 41, 42, 43, 44]

# Old CC 40WTC
# CC_wsi_list =  [6, 8, 11, 12, 13, 14, 39, 52, 54, 55, 72, 108, 111, 116, 122, 124, 130, 131, 137, 138, 143, 144, 169, 170, 171, 175, 178, 180, 183, 184 , 185, 208, 244, 246, 247, 248, 291, 374, 375, 377]
# CC_wsi_list = [3, 41, 71, 91, 123, 135,177, 192, 207, 222]
# CC_wsi_list = [1, 3, 6, 7, 8, 11, 12, 13, 14, 15, 39, 40, 42, 43, 52, 53, 54, 55, 67, 70, 71, 72, 88, 91, 95, 100, 108, 109, 111, 118, 122, 124, 130, 131, 135, 136, 175, 178, 191, 202]

# New CC 40WTC
# CC_wsi_list = [1, 6, 7, 8, 11, 12, 13, 14, 39, 40, 52, 53, 54, 55, 70, 72, 100, 108, 111, 118, 122, 124, 130, 131, 136, 137, 138, 143, 144, 169, 170, 171, 175, 178, 180, 183, 184, 191, 202]
# CC_wsi_list = [2, 41, 69, 90, 110, 123, 134, 177, 190, 201]


# HCC 100WTC
# HCC_wsi_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 48, 50, 51, 52, 54, 55, 56, 58, 59, 60, 62, 63, 64, 66, 67, 68, 70, 71, 73, 74, 75, 77, 78, 79, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 97, 99, 104, 107, 118, 120, 121, 122, 127, 129, 130, 135, 136, 138, 140, 141, 142, 144]
# HCC_wsi_list = [45, 49, 53, 57, 61, 65, 69, 72, 76, 80]

# CC 100WTC
# CC_wsi_list = [1, 3, 6, 7, 8, 11, 12, 13, 14, 15, 39, 40, 42, 43, 52, 53, 54, 55, 67, 70, 71, 72, 88, 91, 95, 100, 108, 109, 111, 118, 122, 124, 130, 131, 135, 136, 137, 138, 143, 144, 145, 167, 168, 169, 170, 171, 173, 174, 175, 178, 179, 180, 183, 184, 185, 189, 191, 192, 202, 204, 206, 207, 208, 215, 217, 222, 223, 224, 225, 226, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 263, 264, 265, 266, 269, 275, 276, 277, 291, 296, 297, 298, 299, 300, 325, 328, 329, 330, 374, 375]
# CC_wsi_list = [2, 41, 69, 90, 110, 123, 134, 177, 190, 201]

# large test set
# HCC_wsi_list = [105, 117, 133, 151, 153, 154, 159, 160, 168, 169, 170, 171, 178, 180, 181, 183, 186, 189, 190, 194]
# CC_wsi_list =  [373, 376, 377, 378, 379, 380, 390, 391, 392, 400, 401, 402, 406, 407, 408, 409, 410, 422, 454, 455]

# Generation Training
# HCC_wsi_list = [104, 107, 109, 111, 120, 121, 122, 129, 131, 132, 135, 139, 141, 142, 145, 146, 149, 150, 153, 156, 159, 161, 162, 164, 165, 166, 168, 169, 171, 175]
HCC_wsi_list = [h+91 for h in HCC_wsi_list]
# CC_wsi_list = [146, 158, 163, 164, 165, 315, 316, 331, 363, 459, 460, 461, 468, 469, 470, 471, 472, 473, 474, 475, 476, 483, 484, 487, 491, 492, 493, 495, 497, 499]


def add_results(file_path, cl, wsi, num_trial, results):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return results  # Skip if file does not exist

    # Read the CSV file
    df = pd.read_csv(file_path)
    cols = ['Accuracy', f'{cl}_TP', f'{cl}_FN', f'{cl}_TN', f'{cl}_FP']

    if set(cols).issubset(df.columns):
        acc = df.iloc[0]['Accuracy']
        tp = df.iloc[0][f'{cl}_TP']
        fn = df.iloc[0][f'{cl}_FN']
        tn = df.iloc[0][f'{cl}_TN']
        fp = df.iloc[0][f'{cl}_FP']

        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else np.nan

        results.append([num_trial, wsi, cl, acc, tp, fn, tn, fp, precision, recall, f1])  # Add trial and WSI ID for reference
    else:
        print(f"Missing columns in {file_path}, {cl}")

    return results

def collect_results(wsi, cl, results):
    if ep == 0:
        condition = f"{wsi}_{num_wsi}WTC_LP{data_num}_{num_class}_class_trial_{num_trial}"
    else:
        condition = f"{wsi}_{num_wsi}WTC_LP{data_num}_{num_class}_class_trial_{num_trial}_for_epoch_{ep}"

    if num_wsi == 1:
        file_path = f"{base_path}/{wsi}/trial_{num_trial}/Metric/{condition}_test_result.csv"
    else:
        file_path = f"{base_path}/trial_{num_trial}/{wsi}/Metric/{condition}_test_result.csv"
        
    results = add_results(file_path, cl, wsi, num_trial, results)

    return results

# Initialize result list
results = []

# Iterate through trials and WSIs
for _wsi in HCC_wsi_list:
    wsi = _wsi
    results = collect_results(wsi, "H", results)
    results = collect_results(wsi, "N", results)
    results = collect_results(wsi, "F", results)

for _wsi in CC_wsi_list:
    wsi = f"1{_wsi:04d}"
    results = collect_results(wsi, "C", results)
    results = collect_results(wsi, "N", results)
    results = collect_results(wsi, "F", results)
            
# --- Convert to DataFrame ---
df = pd.DataFrame(results, columns=['Trial', 'WSI', 'Class', 'Accuracy', 'TP', 'FN', 'TN', 'FP', 'Precision', 'Recall', 'F1'])

# --- Separate tumor (H/C) and normal (N) ---
df_tumor = df[df['Class'].isin(['H', 'C'])].copy()
df_normal = df[df['Class'] == 'N'].copy()
df_fib = df[df['Class'] == 'F'].copy()

# Merge on Trial, WSI, Gen, Condition
df_tumor = df_tumor.rename(
    columns={'TP': 'T_TP', 'FN': 'T_FN', 'TN': 'T_TN', 'FP': 'T_FP', 'Precision': 'T_Precision', 'Recall': 'T_Recall', 'F1': 'T_F1'})

df_normal = df[df['Class'] == 'N'].copy().rename(
    columns={'TP': 'N_TP', 'FN': 'N_FN', 'TN': 'N_TN', 'FP': 'N_FP', 'Precision': 'N_Precision', 'Recall': 'N_Recall', 'F1': 'N_F1'})

df_fib = df[df['Class'] == 'F'].copy().rename(
    columns={'TP': 'F_TP', 'FN': 'F_FN', 'TN': 'F_TN', 'FP': 'F_FP', 'Precision': 'F_Precision', 'Recall': 'F_Recall', 'F1': 'F_F1'})

merged = pd.merge(df_tumor, df_normal[['Trial', 'WSI', 'N_F1', 'N_Precision', 'N_Recall', 'N_TP', 'N_FN', 'N_TN', 'N_FP']], on=['Trial', 'WSI'], how='left')
merged = pd.merge(merged, df_fib[['Trial', 'WSI', 'F_F1', 'F_Precision', 'F_Recall', 'F_TP', 'F_FN', 'F_TN', 'F_FP']], on=['Trial', 'WSI'], how='left')

merged['Macro_F1'] = merged[['T_F1', 'N_F1', 'F_F1']].mean(axis=1)
merged['Macro_Precision'] = merged[['T_Precision', 'N_Precision', 'F_Precision']].mean(axis=1)
merged['Macro_Recall'] = merged[['T_Recall', 'N_Recall', 'F_Recall']].mean(axis=1)

# --- Final columns ---
custom_order = [
    'Trial', 'WSI', 'Accuracy',
    'T_TP', 'T_FN', 'T_TN', 'T_FP', 
    'N_TP', 'N_FN', 'N_TN', 'N_FP', 
    'F_TP', 'F_FN', 'F_TN', 'F_FP',
    'T_Precision', 'N_Precision', 'F_Precision', 'Macro_Precision',
    'T_Recall', 'N_Recall', 'F_Recall', 'Macro_Recall',
    'T_F1', 'N_F1', 'F_F1', 'Macro_F1' 
]

existing_cols = [c for c in custom_order if c in merged.columns]
df_output = merged[existing_cols]
df_output.to_csv(output_file, index=False)

print(f"Processed results saved to {output_file}")
