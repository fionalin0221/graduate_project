import os
import pandas as pd

# Define paths
result_type = "Mix"
num_wsi = 100
data_num = "ALL"
num_trial = 7
num_class = 3

# base_path = f"/workspace/Data/Results/{result_type}_NDPI/{num_wsi}WTC_Result/LP_{data_num}"
# base_path = f"/home/ipmclab-2/project/Results/{result_type}_NDPI/{num_wsi}WTC_Result/LP_{data_num}"
base_path = f"/home/ipmclab/project/Results/{result_type}_NDPI/{num_wsi}WTC_Result/LP_{data_num}"
output_file = f"{base_path}/{num_wsi}WTC_LP{data_num}_trial_{num_trial}_test_results.csv"

# Define trials and WSIs
# HCC_wsi_list = []
# CC_wsi_list = []

# HCC 10WTC
# HCC_wsi_list = [1, 12, 22, 33, 45, 56, 67, 76, 89, 91]
# HCC_wsi_list = [6, 11, 39, 52, 144]
# CC 10WTC
# CC_wsi_list = [72, 108, 111, 116, 122, 124, 130, 131, 137, 138]
# CC_wsi_list = [2, 21, 50, 69, 81]

# Old CC 40WTC
# CC_wsi_list =  [6, 8, 11, 12, 13, 14, 39, 52, 54, 55, 72, 108, 111, 116, 122, 124, 130, 131, 137, 138, 143, 144, 169, 170, 171, 175, 178, 180, 183, 184 , 185, 208, 244, 246, 247, 248, 291, 374, 375, 377]
# CC_wsi_list = [3, 41, 71, 91, 123, 135,177, 192, 207, 222]
# CC_wsi_list = [1, 3, 6, 7, 8, 11, 12, 13, 14, 15, 39, 40, 42, 43, 52, 53, 54, 55, 67, 70, 71, 72, 88, 91, 95, 100, 108, 109, 111, 118, 122, 124, 130, 131, 135, 136, 175, 178, 191, 202]

# New CC 40WTC
# CC_wsi_list = [1, 6, 7, 8, 11, 12, 13, 14, 39, 40, 52, 53, 54, 55, 70, 72, 100, 108, 111, 118, 122, 124, 130, 131, 136, 137, 138, 143, 144, 169, 170, 171, 175, 178, 180, 183, 184, 191, 202]
# CC_wsi_list = [2, 41, 69, 90, 110, 123, 134, 177, 190, 201]

# Mix 100WTC
# HCC_wsi_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 48, 50, 51, 52, 54, 55, 56, 58, 59, 60, 62, 63, 64, 66, 67, 68, 70, 71, 73, 74, 75, 77, 78, 79, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 97, 99, 104, 107, 118, 120, 121, 122, 127, 129, 130, 135, 136, 138, 140, 141, 142, 144]
# CC_wsi_list = [1, 3, 6, 7, 8, 11, 12, 13, 14, 15, 39, 40, 42, 43, 52, 53, 54, 55, 67, 70, 71, 72, 88, 91, 95, 100, 108, 109, 111, 118, 122, 124, 130, 131, 135, 136, 137, 138, 143, 144, 145, 167, 168, 169, 170, 171, 173, 174, 175, 178, 179, 180, 183, 184, 185, 189, 191, 192, 202, 204, 206, 207, 208, 215, 217, 222, 223, 224, 225, 226, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 263, 264, 265, 266, 269, 275, 276, 277, 291, 296, 297, 298, 299, 300, 325, 328, 329, 330, 374, 375]
# HCC_wsi_list = [45, 49, 53, 57, 61, 65, 69, 72, 76, 80]
# CC_wsi_list = [2, 41, 69, 90, 110, 123, 134, 177, 190, 201]

# large test set
HCC_wsi_list = [105, 117, 133, 151, 153, 154, 159, 160, 168, 169, 170, 171, 178, 180, 181, 183, 186, 189, 190, 194] #, 147, 148, 195, 198, 200, 202, 211, 212, 213, 220, 222, 223, 226, 230, 232, 233, 236, 237, 240, 241]
CC_wsi_list =  [373, 376, 377, 378, 379, 380, 390, 391, 392, 400, 401, 402, 406, 407, 408, 409, 410, 422, 454, 455] #, 146, 158, 163, 164, 165, 363, 459, 460, 461, 468, 469, 470, 471, 472, 474, 475, 476, 483, 484]


def add_results(file_path, cl, wsi, num_trial, results):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return results  # Skip if file does not exist

    # Read the CSV file
    df = pd.read_csv(file_path)

    if {'Accuracy', f'{cl}_TP', f'{cl}_FN'}.issubset(df.columns):
        row = df.iloc[0][['Accuracy', f'{cl}_TP', f'{cl}_FN']]
        results.append([num_trial, wsi, cl] + row.tolist())  # Add trial and WSI ID for reference
    else:
        print(f"Missing columns in {file_path}")

    return results

def collect_results(wsi, cl, results):
    if len(str(wsi)) == 5:
        _wsi = int(wsi[-3:])
        if num_wsi == 1:
            file_path = f"{base_path}/{wsi}/trial_{num_trial}/Metric/{wsi}_{num_wsi}WTC_LP{data_num}_{num_class}_class_trial_{num_trial}_test_result.csv"
        else:
            file_path = f"{base_path}/trial_{num_trial}/{wsi}/Metric/{wsi}_{num_wsi}WTC_LP{data_num}_{num_class}_class_trial_{num_trial}_for_epoch_9_test_result.csv"
    else:
        if num_wsi == 1:
            file_path = f"{base_path}/{wsi}/trial_{num_trial}/Metric/{wsi}_{num_wsi}WTC_LP{data_num}_{num_class}_class_trial_{num_trial}_test_result.csv"
        else:
            file_path = f"{base_path}/trial_{num_trial}/{wsi}/Metric/{wsi}_{num_wsi}WTC_LP{data_num}_{num_class}_class_trial_{num_trial}_for_epoch_9_test_result.csv"
        
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
df = pd.DataFrame(results, columns=['Trial', 'WSI', 'Class', 'Accuracy', 'TP', 'FN'])

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
df_output = merged[['Trial', 'WSI', 'Accuracy', 'TP', 'FN', 'TN', 'FP']]
df_output.to_csv(output_file, index=False)

print(f"Processed results saved to {output_file}")
