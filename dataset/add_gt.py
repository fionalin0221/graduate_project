import pandas as pd

wsi = 2

gt_df = pd.read_csv(f"/workspace/Data/Results/CC_NDPI/Data_Info/{wsi}/1{wsi:04d}_patch_in_region_filter_2_v2.csv")
gt_dict = dict(zip(gt_df['file_name'], gt_df['label']))

# selected, ideal

for num_trial in range(1, 2):
    for gen in range(1, 5):
        # df = pd.read_csv(f"/workspace/Data/Results/Mix_NDPI/Generation_Training/40WTC_LP_6400/1{wsi:04d}/trial_{num_trial}/Data/1{wsi:04d}_Gen{gen}_ND_zscore_selected_patches_by_Gen{gen-1}.csv")
        df = pd.read_csv(f"/workspace/Data/Results/Mix_NDPI/Generation_Training/40WTC_LP_6400/1{wsi:04d}/trial_{num_trial}/Data/1{wsi:04d}_Gen{gen}_ND_zscore_ideal_patches_by_Gen{gen-1}.csv")
        # df = pd.read_csv(f"/workspace/Data/Results/Mix_NDPI/Generation_Training/40WTC_LP_6400/1{wsi:04d}/trial_{num_trial}/Data/Gen{gen}_ND_zscore_ideal_patches_by_Gen{gen-1}_train.csv")
        # origin_pred_df = pd.read_csv(f"/workspace/Data/Results/Mix_NDPI/Generation_Training/40WTC_LP_6400/1{wsi:04d}/trial_{num_trial}/Metric/1{wsi:04d}_Gen{gen-1}_ND_zscore_ideal_patches_by_Gen{gen-2}_labels_predictions.csv")
        # origin_pred_df = pd.read_csv(f"/workspace/Data/Results/Mix_NDPI/Generation_Training/40WTC_LP_6400/1{wsi:04d}/trial_{num_trial}/Metric/1{wsi:04d}_3_class_labels_predictions.csv")
        # origin_dict = dict(zip(origin_pred_df['file_name'], origin_pred_df['pred_label']))

        # Rename 'label' to 'predict_label'
        df.rename(columns={'label': 'pred_label'}, inplace=True)
        # df.rename(columns={'label': 'flipped_label'}, inplace=True)

        # Add ground truth column using lookup
        df['gt_label'] = df['file_name'].map(gt_dict)
        # df['pred_label'] = df['file_name'].map(origin_dict)

        # Keep only needed columns
        # df = df[['file_name', 'flipped_label', 'pred_label', 'gt_label']]
        df = df[['file_name', 'pred_label', 'gt_label']]
        # df = df[['file_name', 'pred_label', 'gt_label', 'augment']]

        # Save
        df.to_csv(
            # f"/workspace/Data/Results/Mix_NDPI/Generation_Training/40WTC_LP_6400/1{wsi:04d}/trial_{num_trial}/1{wsi:04d}_Gen{gen}_ND_zscore_selected_patches_by_Gen{gen-1}_with_gt.csv",
            f"/workspace/Data/Results/Mix_NDPI/Generation_Training/40WTC_LP_6400/1{wsi:04d}/trial_{num_trial}/1{wsi:04d}_Gen{gen}_ND_zscore_ideal_patches_by_Gen{gen-1}_with_gt.csv",
            # f"/workspace/Data/Results/Mix_NDPI/Generation_Training/40WTC_LP_6400/1{wsi:04d}/trial_{num_trial}/Gen{gen}_ND_zscore_ideal_patches_by_Gen{gen-1}_train_with_gt.csv",
            index=False
        )