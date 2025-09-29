import pandas as pd
import numpy as np
from tqdm import tqdm

wsi = 2
base_path = "/home/ipmclab-2/project"

gt_df = pd.read_csv(f"{base_path}/Results/CC_NDPI/Data_Info/{wsi}/1{wsi:04d}_patch_in_region_filter_2_v2.csv")
gt_dict = dict(zip(gt_df['file_name'], gt_df['label']))

# selected, ideal

for num_trial in range(3, 4):
    for gen in range(1, 5):
        df = pd.read_csv(f"{base_path}/Results/Mix_NDPI/Generation_Training/40WTC_LP_6400/1{wsi:04d}/trial_{num_trial}/Data/1{wsi:04d}_Gen{gen}_ND_zscore_selected_patches_by_Gen{gen-1}.csv")
        if gen == 1:
            origin_pred_df = pd.read_csv(f"{base_path}/Results/Mix_NDPI/Generation_Training/40WTC_LP_6400/1{wsi:04d}/trial_{num_trial}/TI/1{wsi:04d}_3_class_all_patches_filter_v2_TI.csv")
        else:
            origin_pred_df = pd.read_csv(f"{base_path}/Results/Mix_NDPI/Generation_Training/40WTC_LP_6400/1{wsi:04d}/trial_{num_trial}/TI/1{wsi:04d}_Gen{gen-1}_ND_zscore_selected_patches_by_Gen{gen-2}_all_patches_filter_v2_TI.csv")

        # Add empty pred_label column
        origin_pred_df["pred_label"] = None

        # Iterate each row
        for idx, row in tqdm(origin_pred_df.iterrows()):
            preds = [row["N_pred"], row["H_pred"], row["C_pred"]]
            over_threshold = [i for i, p in enumerate(preds) if p > 0.5]

            if len(over_threshold) == 1:  # exactly one above 0.5
                if over_threshold[0] == 0:
                    origin_pred_df.at[idx, "pred_label"] = "N"
                elif over_threshold[0] == 1:
                    origin_pred_df.at[idx, "pred_label"] = "H"
                elif over_threshold[0] == 2:
                    origin_pred_df.at[idx, "pred_label"] = "C"
            # else: keep as None → will appear blank in DataFrame/CSV

        origin_dict = dict(zip(origin_pred_df['file_name'], origin_pred_df['pred_label']))

        # Rename 'label' to 'predict_label'
        df.rename(columns={'label': 'flipped_label'}, inplace=True)

        # Add ground truth column using lookup
        df['gt_label'] = df['file_name'].map(gt_dict)
        df['pred_label'] = df['file_name'].map(origin_dict)

        # Keep only needed columns
        df = df[['file_name', 'flipped_label', 'pred_label', 'gt_label']]

        # Save
        df.to_csv(
            f"{base_path}/Results/Mix_NDPI/Generation_Training/40WTC_LP_6400/1{wsi:04d}/trial_{num_trial}/1{wsi:04d}_Gen{gen}_ND_zscore_selected_patches_by_Gen{gen-1}_with_gt.csv",
            index=False
        )

        df = pd.read_csv(f"{base_path}/Results/Mix_NDPI/Generation_Training/40WTC_LP_6400/1{wsi:04d}/trial_{num_trial}/Data/Gen{gen}_ND_zscore_selected_patches_by_Gen{gen-1}_train.csv")

        # Rename 'label' to 'predict_label'
        df.rename(columns={'label': 'pred_label'}, inplace=True)

        # Add ground truth column using lookup
        df['gt_label'] = df['file_name'].map(gt_dict)

        # Keep only needed columns
        df = df[['file_name', 'pred_label', 'gt_label', 'augment']]

        # Save
        df.to_csv(
            f"{base_path}/Results/Mix_NDPI/Generation_Training/40WTC_LP_6400/1{wsi:04d}/trial_{num_trial}/Gen{gen}_ND_zscore_selected_patches_by_Gen{gen-1}_train_with_gt.csv",
            index=False
        )