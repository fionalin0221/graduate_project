import os
import pandas as pd

# Define paths
base_path = "/workspace/Data/Results/CC_NDPI/40WTC_Result/LP_6400"
output_file = "/workspace/Data/Results/CC_NDPI/40WTC_Result/LP_6400/transfer_model_40_finetune_tati_results.csv"

# Define trials and WSIs
num_trials = list(range(2, 3))  # 11 to 19
# wsi_list = [3, 41, 71, 91, 123, 135,177, 192, 207, 222]
wsi_list =  [6, 8, 11, 12, 13, 14, 39, 52, 54, 55, 72, 108, 111, 116, 122, 124, 130, 131, 137, 138, 143, 144, 169, 170, 171, 175, 178, 180, 183, 184 , 185, 208, 244, 246, 247, 248, 291, 374, 375, 377]


# Initialize result list
results = []

# Iterate through trials and WSIs
for num_trial in num_trials:
    for _wsi in wsi_list:
        wsi = f"1{_wsi:04d}"
        file_path = f"{base_path}/trial_{num_trial}/{wsi}/Metric/{wsi}_Model_40_finetune_40WTC_LP6400_2_class_trial_{num_trial}_test_result.csv"

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue  # Skip if file does not exist

        # Read the CSV file
        df = pd.read_csv(file_path)

        # Extract required columns
        if {'Accuracy', 'C_TP', 'C_FN', 'C_TN', 'C_FP'}.issubset(df.columns):
            row = df.iloc[0][['Accuracy', 'C_TP', 'C_FN', 'C_TN', 'C_FP']]
            results.append([num_trial, wsi] + row.tolist())  # Add trial and WSI ID for reference
        else:
            print(f"Missing columns in {file_path}")

# Convert results into DataFrame and save to CSV
df_output = pd.DataFrame(results, columns=['Trial', 'WSI', 'Accuracy', 'TP', 'FN', 'TN', 'FP'])
df_output.to_csv(output_file, index=False)

print(f"Processed results saved to {output_file}")
