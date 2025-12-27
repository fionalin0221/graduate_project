import pandas as pd
import os

def merge_labels(base_path, wsi, condition):
    
    file_path = f"{base_path}/{wsi}/Metric/{condition}_first_stage_labels_predictions.csv"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    df_stage1 = pd.read_csv(f"{base_path}/{wsi}/Metric/{condition}_first_stage_labels_predictions.csv")
    df_stage2 = pd.read_csv(f"{base_path}/{wsi}/Metric/{condition}_second_stage_labels_predictions.csv")

    df_stage2 = df_stage2.rename(columns={
        "pred_label": "pred_label_stage2"
    })

    df_merge = df_stage1.merge(
        df_stage2[["file_name", "pred_label_stage2"]],
        on="file_name",
        how="left"
    )

    df_merge["final_pred_label"] = df_merge.apply(
        lambda row: row["pred_label_stage2"]
        if row["pred_label"] == "others"
        else row["pred_label"],
        axis=1
    )

    final_df = df_merge[["file_name", "true_label", "final_pred_label"]]
    final_df = final_df.rename(columns={
        "final_pred_label": "pred_label"
    })
    final_df.to_csv(f"{base_path}/{wsi}/Metric/{condition}_labels_predictions.csv", index=False)
    return final_df

def merge_TI(base_path, wsi, condition, classes, target_class):
    file_path = f"{base_path}/{wsi}/TI/{condition}_first_stage_patch_in_region_filter_2_v2_TI.csv"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    df_stage1 = pd.read_csv(f"{base_path}/{wsi}/TI/{condition}_first_stage_patch_in_region_filter_2_v2_TI.csv")
    df_stage2 = pd.read_csv(f"{base_path}/{wsi}/TI/{condition}_second_stage_patch_in_region_filter_2_v2_TI.csv")

    stage_2_classes = classes.copy()
    stage_2_classes.remove(target_class)

    cols = ["file_name"] + [f"{cl}_pred" for cl in stage_2_classes]
    df_merge = df_stage1.merge(
        df_stage2[cols],
        on="file_name",
        how="left"
    )
    df_merge.to_csv(f"{base_path}/{wsi}/TI/{condition}_patch_in_region_filter_2_v2_TI.csv", index=False)
    
    return df_merge

def main():
    result_type = "Mix"
    num_wsi = 40
    data_num = 3200
    num_trial = 11
    num_class = 3

    base_path = f"/home/ipmclab/project/Results/{result_type}_NDPI/{num_wsi}WTC_Result/LP_{data_num}/trial_{num_trial}"
    HCC_wsis = [14, 26, 42, 60, 62, 63, 68, 69, 77, 78, 79, 80, 87, 89, 90, 92, 95, 98, 99, 103]
    CC_wsis = [373, 376, 377, 378, 379, 380, 390, 391, 392, 400, 401, 402, 406, 407, 408, 409, 410, 422, 454, 455]

    for _wsi in HCC_wsis:
        wsi = _wsi + 91
        condition = f"{wsi}_{num_wsi}WTC_LP{data_num}_{num_class}_class_trial_{num_trial}"
        # merge_labels(base_path, wsi, condition)
        merge_TI(base_path, wsi, condition, classes = ['N','H','C'], target_class = 'C')

    for _wsi in CC_wsis:
        wsi = f"1{_wsi:04d}"
        condition = f"{wsi}_{num_wsi}WTC_LP{data_num}_{num_class}_class_trial_{num_trial}"
        # merge_labels(base_path, wsi, condition)
        merge_TI(base_path, wsi, condition, classes = ['N','H','C'], target_class = 'C')

if __name__ == "__main__":
    main()