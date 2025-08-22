import os
import pandas as pd

# Define paths
# base_dir = "/workspace/Data/Results/HCC_NDPI/Data_Info"  # Change this to your directory containing CSV files

cl = "H"
if cl == "H":
    base_dir = "/home/ipmclab-2/project/Results/HCC_NDPI/Data_Info"
    output_file = "100WTC_HCC_test_WSI_patches.csv"  # Output file
elif cl == "C":
    base_dir = "/home/ipmclab-2/project/Results/CC_NDPI/Data_Info"
    output_file = "100WTC_CC_WSI_patches.csv"  # Output file

# Store results
results = []
# wsis = [1, 3, 6, 7, 8, 11, 12, 13, 14, 15, 39, 40, 42, 43, 52, 53, 54, 55, 67, 70, 71, 72, 88, 91, 95, 100, 108, 109, 111, 118, 122, 124, 130, 131, 135, 136, 137, 138, 143, 144, 145, 167, 168, 169, 170, 171, 173, 174, 175, 178, 179, 180, 183, 184, 185, 189, 191, 192, 202, 204, 206, 207, 208, 215, 217, 222, 223, 224, 225, 226, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 263, 264, 265, 266, 269, 275, 276, 277, 291, 296, 297, 298, 299, 300, 325, 328, 329, 330, 374, 375]
# wsis = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 48, 50, 51, 52, 54, 55, 56, 58, 59, 60, 62, 63, 64, 66, 67, 68, 70, 71, 73, 74, 75, 77, 78, 79, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 97, 99, 104, 107, 118, 120 ,121 , 122, 127, 129, 130, 135, 136, 138, 140, 141, 142, 144]
wsis = [45, 49, 53, 57, 61, 65, 69, 72, 76, 80]
# wsis = [2, 41, 69, 90, 110, 123, 134, 177, 190, 201]
# wsis = [1, 3, 6, 7, 8, 11, 12, 13, 14, 15, 39, 40, 42, 43, 52, 53, 54, 55, 67, 70, 71, 72, 88, 91, 95, 100, 108, 109, 111, 118, 122, 124, 130, 131, 135, 136, 175, 178, 191, 202]

# Loop through directories
for num in wsis:  # Adjust range if needed
    dir_path = os.path.join(base_dir, str(num))  # Directory path
    if cl == "H":
        file_name = f"{num}_patch_in_region_filter_2_v2.csv"
    elif cl == "C":
        file_name = f"1{num:04d}_patch_in_region_filter_2_v2.csv"
    
    file_path = os.path.join(dir_path, file_name)
    print(file_path)
    
    # Check if file exists
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)  # Read CSV

        # Count occurrences of "C" and "N"
        count_C = (df['label'] == cl).sum()
        count_N = (df['label'] == 'N').sum()

        # Append results
        results.append({"WSI": num, f"count_{cl}": count_C, "count_N": count_N})

# Save to CSV
df_results = pd.DataFrame(results)
df_results.to_csv(output_file, index=False)

print(f"Label counts saved to {output_file}")
