import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import cv2
from tqdm import tqdm

num_wsi = 100
data_num = "ALL"
num_trial = 1

base_path_hcc = f'/home/ipmclab/project/Results/HCC_NDPI/{num_wsi}WTC_Result/LP_{data_num}/trial_{num_trial}'
base_path_cc = f'/home/ipmclab/project/Results/CC_NDPI/{num_wsi}WTC_Result/LP_{data_num}/trial_{num_trial}'
base_path_mix = f'/home/ipmclab/project/Results/Mix_NDPI/{num_wsi}WTC_Result/LP_3200/trial_{num_trial}'
output_file = f"{base_path_mix}/{num_wsi}WTC_LP3200_trial_{num_trial}_test_results_combined.csv"

HCC_wsi_list = [105, 117, 133, 151, 153, 154, 159, 160, 168, 169, 170, 171, 178, 180, 181, 183, 186, 189, 190, 194]
CC_wsi_list =  [373, 376, 377, 378, 379, 380, 390, 391, 392, 400, 401, 402, 406, 407, 408, 409, 410, 422, 454, 455]

# def apply_ensemble_rules(row):
#     p1 = row['pred_label_1']
#     p2 = row['pred_label_2']
#     if p1 == 'N' and p2 == 'N':
#         return 'Normal'
#     elif (p1 == 'C' and p2 == 'H') or (p1 == 'H' and p2 == 'C'):
#         return 'Cancer'
#     else:
#         return 'unknown'

def apply_ensemble_rules(row):
    p1 = row['pred_label_1']
    p2 = row['pred_label_2']
    p3 = row['pred_label_3']
    
    if p1 == 'N' and p2 == 'N':
        return 'N'
    else:
        return p3

def define_cm(final_df):
    df = final_df.copy()
    # df['true_binary'] = final_df['true_label'].replace({'C': 'Cancer', 'H': 'Cancer', 'N': 'Normal'})

    cm = pd.crosstab(
        df['true_label'], 
        df['ensemble_result'],
        rownames=['Ground Truth'], 
        colnames=['Ensemble Prediction']
    )

    # cols = [c for c in ['Cancer', 'Normal', 'unknown'] if c in cm.columns]
    cm = cm.reindex(index=['N', 'H', 'C'], columns=['N', 'H', 'C', 'unknown']).fillna(0).astype(int)

    cm_flattened = cm.stack()
    flattened_data = {}
    for (true_lab, pred_lab), value in cm_flattened.items():
        column_name = f"True_{true_lab}_Pred_{pred_lab}"
        flattened_data[column_name] = [value]

    return flattened_data

def process(wsi, cl):
    file_path_hcc = f"{base_path_hcc}/{wsi}/Metric/{wsi}_{num_wsi}WTC_LP{data_num}_2_class_trial_{num_trial}_for_epoch_18_labels_predictions.csv"
    file_path_cc = f"{base_path_cc}/{wsi}/Metric/{wsi}_{num_wsi}WTC_LP{data_num}_2_class_trial_{num_trial}_for_epoch_18_labels_predictions.csv"
    file_path_mix = f"{base_path_mix}/{wsi}/Metric/{wsi}_{num_wsi}WTC_LP3200_3_class_trial_{num_trial}_labels_predictions.csv"
    df_hcc = pd.read_csv(file_path_hcc)
    df_cc = pd.read_csv(file_path_cc)
    df_mix = pd.read_csv(file_path_mix)

    if cl == 'H':
        merged_df = pd.merge(
            df_hcc[['file_name', 'true_label', 'pred_label']], 
            df_cc[['file_name', 'pred_label']], 
            on='file_name', 
            suffixes=('_1', '_2')
        ).merge(
            df_mix[['file_name', 'pred_label']],
            on='file_name'
        ).rename(columns={'pred_label': 'pred_label_3'})
    else:
        merged_df = pd.merge(
            df_cc[['file_name', 'true_label', 'pred_label']], 
            df_hcc[['file_name', 'pred_label']], 
            on='file_name', 
            suffixes=('_1', '_2')
        ).merge(
            df_mix[['file_name', 'pred_label']],
            on='file_name'
        ).rename(columns={'pred_label': 'pred_label_3'})
    

    merged_df['ensemble_result'] = merged_df.apply(apply_ensemble_rules, axis=1)
    final_df = merged_df[['file_name', 'true_label', 'ensemble_result']]
    final_df.to_csv(f'{base_path_mix}/Metric/{wsi}_{num_wsi}WTC_LP{data_num}_3_class_trial_{num_trial}_labels_predictions_combined.csv', index=False)

    flattened_data = define_cm(final_df)
    df_flat = pd.DataFrame(flattened_data)

    df_flat.insert(0, 'WSI', wsi)
    df_flat.insert(1, 'Trial', num_trial)

    df_flat.to_csv(f'{base_path_hcc}/{wsi}/Metric/{wsi}_{num_wsi}WTC_LP{data_num}_3_class_trial_{num_trial}_confusion_matrix_combined.csv', index=False)

    return df_flat

all_results = []

for wsi in HCC_wsi_list:
    result = process(wsi, 'H')
    all_results.append(result)
for wsi in CC_wsi_list:
    wsi = f"1{wsi:04d}"
    result = process(wsi, 'C')
    all_results.append(result)

final_summary_df = pd.concat(all_results, ignore_index=True)
final_summary_df.to_csv(output_file, index=False)