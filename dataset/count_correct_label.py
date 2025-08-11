import json
import pandas as pd
import os

gt_path = "/home/ipmclab/project/Results/CC_NDPI/Data_Info/90/10090_patch_in_region_filter_2_v2.csv"
gt_data = pd.read_csv(gt_path)
gt_dict = dict(zip(gt_data['file_name'], gt_data['label']))

base_path = "/home/ipmclab/project/Results/Mix_NDPI/Generation_Training/40WTC_LP_6400/10090"

classes = ["N", "C"]  # fixed classes
results = []

for trial in range(1, 7):
    for gen in range(1, 5):
        # 10090_Gen{gen}_ND_zscore_ideal_patches_by_Gen{gen-1}
        # Gen{gen}_ND_zscore_ideal_patches_by_Gen{gen-1}_train
        pred_path = f"{base_path}/trial_{trial}/Data/Gen{gen}_ND_zscore_ideal_patches_by_Gen{gen-1}_train.csv"
        train_data = pd.read_csv(pred_path)
        
        correct = 0
        incorrect = 0
        total = 0

        class_stats = {cls: {"Correct": 0, "Incorrect": 0, "Total": 0} for cls in classes}

        file_names = train_data['file_name'].tolist()
        pseudo_labels = train_data['label'].tolist()

        for fname, pseudo_label in zip(file_names, pseudo_labels):
            if fname not in gt_dict:
                print(f"Warning: {fname} not found in ground truth")
                continue

            gt_label = gt_dict[fname]
            if pseudo_label == gt_label:
                correct += 1
                class_stats[gt_label]["Correct"] += 1
            else:
                incorrect += 1
                class_stats[gt_label]["Incorrect"] += 1
            total += 1
            class_stats[gt_label]["Total"] += 1

        accuracy = 100 * correct / total if total > 0 else 0.0

        print(f"Trial {trial} Gen {gen}:")
        print(f"  Total patches: {total}")
        print(f"  Correct pseudo labels: {correct}")
        print(f"  Incorrect pseudo labels: {incorrect}")
        print(f"  Accuracy: {accuracy:2f}%")

        row = {
            "Trial": trial,
            "Gen": gen,
            "Total": total,
            "Correct": correct,
            "Incorrect": incorrect,
        }

        # add per-class stats into the same row
        for cls, stats in sorted(class_stats.items(), key=lambda x: x[0]):
            row[f"Class_{cls}_Total"] = stats["Total"]
            row[f"Class_{cls}_Correct"] = stats["Correct"]
            row[f"Class_{cls}_Incorrect"] = stats["Incorrect"]
            print(cls, stats["Total"],stats["Correct"], stats["Incorrect"])

        results.append(row)

df = pd.DataFrame(results)
output_csv = os.path.join(base_path, "10090_train_pseudo_label_summary.csv")
df.to_csv(output_csv, index=False)