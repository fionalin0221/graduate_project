import os
import pandas as pd

# Define paths
base_dir = "/workspace/Data/Results/CC_NDPI/Data_Info"  # Change this to your directory containing CSV files
output_file = "WSI_patches.csv"  # Output file

# Store results
results = []

# Loop through directories
for num in range(1, 250):  # Adjust range if needed
    dir_path = os.path.join(base_dir, str(num))  # Directory path
    file_name = f"1{num:04d}_patch_in_region_filter_2_v2.csv"
    file_path = os.path.join(dir_path, file_name)
    print(file_path)
    
    # Check if file exists
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)  # Read CSV

        # Count occurrences of "C" and "N"
        count_C = (df['label'] == 'C').sum()
        count_N = (df['label'] == 'N').sum()

        # Append results
        results.append({"WSI": num, "count_C": count_C, "count_N": count_N})

# Save to CSV
df_results = pd.DataFrame(results)
df_results.to_csv(output_file, index=False)

print(f"Label counts saved to {output_file}")
