import os
import pandas as pd

# Define paths
base_path = "/workspace/Data/Results/CC_NDPI/5WTC_Result/LP_2560"
output_file = "/workspace/Data/Results/CC_NDPI/5WTC_Result/LP_2560/summary_results.csv"

# Define trials and WSIs
num_trials = list(range(21, 30))  # 11 to 19
wsi_list = [10011, 10039, 10108, 10143, 10169, 10215, 10217, 10222]

# Initialize result list
results = []

# Iterate through trials and WSIs
for num_trial in num_trials:
    for wsi in wsi_list:
        file_path = f"{base_path}/trial_{num_trial}/{wsi}/Metric/{wsi}_5WTC_LP2560_2_class_trial_{num_trial}_test_result.csv"

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
