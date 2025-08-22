import os
import pandas as pd

# Define paths
result_type = "Mix"
num_wsi = 100
data_num = 3200
num_trial = 1
gen = None

# base_path = f"/workspace/Data/Results/{result_type}_NDPI/{num_wsi}WTC_Result/LP_{data_num}"
# output_file = f"{base_path}/{num_wsi}WTC_LP{data_num}_trial_{num_trial}_results.csv"
base_path = f"/home/ipmclab-2/project/Results/{result_type}_NDPI/{num_wsi}WTC_Result/LP_{data_num}"
output_file = f"{base_path}/{num_wsi}WTC_LP{data_num}_trial_{num_trial}_tati_results_new_.csv"
# base_path = f"/home/ipmclab/project/Results/{result_type}_NDPI/Generation_Training/{num_wsi}WTC_LP_{data_num}"
# output_file = f"{base_path}/{num_wsi}WTC_LP{data_num}_trial_{num_trial}_generation_results.csv"

# Define trials and WSIs
# HCC_wsi_list = [195, 275, 298, 314]

# Old CC 40WTC
# CC_wsi_list =  [6, 8, 11, 12, 13, 14, 39, 52, 54, 55, 72, 108, 111, 116, 122, 124, 130, 131, 137, 138, 143, 144, 169, 170, 171, 175, 178, 180, 183, 184 , 185, 208, 244, 246, 247, 248, 291, 374, 375, 377]
# CC_wsi_list = [3, 41, 71, 91, 123, 135,177, 192, 207, 222]
# New CC 40WTC
# CC_wsi_list = [1, 3, 6, 7, 8, 11, 12, 13, 14, 15, 39, 40, 42, 43, 52, 53, 54, 55, 67, 70, 71, 72, 88, 91, 95, 100, 108, 109, 111, 118, 122, 124, 130, 131, 135, 136, 175, 178, 191, 202]
# CC_wsi_list = [2, 41, 69, 90, 110, 123, 134, 177, 190, 201]

# HCC 10WTC
# HCC_wsi_list = [1, 12, 22, 33, 45, 56, 67, 76, 89, 91]
# HCC_wsi_list = [72, 108, 111, 116, 122, 124, 130, 131, 137, 138]
# HCC_wsi_list = [6, 11, 39, 52, 144]
# HCC_wsi_list = [2, 21, 50, 69, 81]

# Mix 100WTC
HCC_wsi_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 48, 50, 51, 52, 54, 55, 56, 58, 59, 60, 62, 63, 64, 66, 67, 68, 70, 71, 73, 74, 75, 77, 78, 79, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 97, 99, 104, 107, 118, 120, 121, 122, 127, 129, 130, 135, 136, 138, 140, 141, 142, 144]
CC_wsi_list = [1, 3, 6, 7, 8, 11, 12, 13, 14, 15, 39, 40, 42, 43, 52, 53, 54, 55, 67, 70, 71, 72, 88, 91, 95, 100, 108, 109, 111, 118, 122, 124, 130, 131, 135, 136, 137, 138, 143, 144, 145, 167, 168, 169, 170, 171, 173, 174, 175, 178, 179, 180, 183, 184, 185, 189, 191, 192, 202, 204, 206, 207, 208, 215, 217, 222, 223, 224, 225, 226, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 263, 264, 265, 266, 269, 275, 276, 277, 291, 296, 297, 298, 299, 300, 325, 328, 329, 330, 374, 375]
# HCC_wsi_list = [45, 49, 53, 57, 61, 65, 69, 72, 76, 80]
# CC_wsi_list = [2, 41, 69, 90, 110, 123, 134, 177, 190, 201]

def add_results(file_path, cl, wsi, num_trial, results):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return results  # Skip if file does not exist

    # Read the CSV file
    df = pd.read_csv(file_path)

    # Extract required columns
    # if {'Accuracy', f'{cl}_TP', f'{cl}_FN', f'{cl}_TN', f'{cl}_FP'}.issubset(df.columns):
    #     row = df.iloc[0][['Accuracy', f'{cl}_TP', f'{cl}_FN', f'{cl}_TN', f'{cl}_FP']]
    #     results.append([num_trial, wsi] + row.tolist())  # Add trial and WSI ID for reference
    # else:
    #     print(f"Missing columns in {file_path}")
    if {'Accuracy', f'{cl}_TP', f'{cl}_FN'}.issubset(df.columns):
        row = df.iloc[0][['Accuracy', f'{cl}_TP', f'{cl}_FN']]
        results.append([num_trial, wsi, cl] + row.tolist())  # Add trial and WSI ID for reference
    else:
        print(f"Missing columns in {file_path}")

    return results

def collect_results(wsi, cl, results):
    if not gen:
        # file_path = f"{base_path}/trial_{num_trial}/{wsi}/Metric/{wsi}_{num_wsi}WTC_LP{data_num}_2_class_trial_{num_trial}_test_result.csv"
        # file_path = f"{base_path}/{wsi}/trial_{num_trial}/Metric/{wsi}_1WTC_LP{data_num}_3_class_trial_{num_trial}_test_result.csv"
        file_path = f"{base_path}/trial_{num_trial}/Metric/{wsi}_{num_wsi}WTC_LP{data_num}_3_class_trial_{num_trial}_test_result.csv"
        results = add_results(file_path, cl, wsi, num_trial, results)

    else:
        for g in range(1, gen):
            file_path = f"{base_path}/{wsi}/trial_{num_trial}/Metric/{wsi}_Gen{g}_ND_zscore_ideal_patches_by_Gen{g-1}_test_result.csv"
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
            
# Convert results into DataFrame and save to CSV
df_output = pd.DataFrame(results, columns=['Trial', 'WSI', 'Class', 'Accuracy', 'TP', 'FN'])
df_output.to_csv(output_file, index=False)

print(f"Processed results saved to {output_file}")
