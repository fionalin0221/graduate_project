import pandas as pd
import os
import shutil

wsi_list = [1, 6, 8, 13, 16, 27, 29 ,30 , 31, 36,  38, 39, 44, 45, 47, 49, 50, 51, 53]
base_path = f'/workspace/Data/Results/HCC_NDPI/1WTC_Result/LP_6400'

for wsi in wsi_list:
    origin_path = f'{base_path}/{wsi}/trial_1'
    save_path = f'{base_path}/{wsi+91}/trial_1'

    os.makedirs(f"{save_path}", exist_ok=True)
    os.makedirs(f"{save_path}/Model", exist_ok=True)
    os.makedirs(f"{save_path}/Metric", exist_ok=True)
    os.makedirs(f"{save_path}/Loss", exist_ok=True)
    os.makedirs(f"{save_path}/TI", exist_ok=True)
    os.makedirs(f"{save_path}/Data", exist_ok=True)

    # shutil.rmtree(f'{save_path}/Model')
    # shutil.rmtree(f'{save_path}/Metric')
    # shutil.rmtree(f'{save_path}/Loss')
    # shutil.rmtree(f'{save_path}/TI')
    # shutil.rmtree(f'{save_path}/Data')

    # Define source and destination paths
    src = f'{origin_path}/Model/{wsi+91}_1WTC_LP6400_2_class_trial_1_Model.ckpt'
    dst = f'{save_path}/Model'  # Can be a folder or a new file path
    # Move the file
    shutil.move(src, dst)

    src = f'{origin_path}/Data/{wsi+91}_1WTC_LP6400_2_class_trial_1_train.csv'
    dst = f'{save_path}/Data'
    shutil.move(src, dst)
    src = f'{origin_path}/Data/{wsi+91}_1WTC_LP6400_2_class_trial_1_valid.csv'
    shutil.move(src, dst)

    src = f'{origin_path}/Loss/{wsi+91}_1WTC_LP6400_2_class_trial_1_epoch_log.csv'
    dst = f'{save_path}/Loss'
    shutil.move(src, dst)
    src = f'{origin_path}/Loss/{wsi+91}_1WTC_LP6400_2_class_trial_1_log.yaml'
    shutil.move(src, dst)
    src = f'{origin_path}/Loss/loss_and_accuracy_curve.png'
    shutil.move(src, dst)

    src = f'{origin_path}/Metric/{wsi+91}_1WTC_LP6400_2_class_trial_1_confusion_matrix.png'
    dst = f'{save_path}/Metric'
    shutil.move(src, dst)
    src = f'{origin_path}/Metric/{wsi+91}_1WTC_LP6400_2_class_trial_1_labels_predictions.csv'
    shutil.move(src, dst)
    src = f'{origin_path}/Metric/{wsi+91}_1WTC_LP6400_2_class_trial_1_pred_score.csv'
    shutil.move(src, dst)
    src = f'{origin_path}/Metric/{wsi+91}_1WTC_LP6400_2_class_trial_1_test_result.csv'
    shutil.move(src, dst)
