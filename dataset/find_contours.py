import os
import numpy as np
import random
import yaml
import cv2
import re
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.path as mpath
from matplotlib.colors import ListedColormap
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler

pts_ratio = 448

def contours_processing(contours, forImage=False):
    regions = []
    for contour in contours:
        contour = contour.reshape(-1, 2)
        if forImage:
            larger_contour = []
            for cnt in contour:
                larger_contour.append(cnt * 4)
            regions.append(np.array(larger_contour))
        else:
            regions.append(contour)
    return regions

def find_contours_of_connected_components(label, sorted_pts, area_thresh):

    ### Make HCC or Normal components regions ###
    x_max = np.max(sorted_pts[:, 0])
    y_max = np.max(sorted_pts[:, 1])
    patches_labels = np.zeros((y_max+1, x_max+1), np.uint8)  # for connected-components of HCC/Normal patches
    
    if label == 'N':
        cate_pts = sorted_pts[sorted_pts[:, 2] == 0]
    if label == 'H':
        cate_pts = sorted_pts[sorted_pts[:, 2] == 1]
    if label == 'C':
        cate_pts = sorted_pts[sorted_pts[:, 2] == 2]

    for pts in cate_pts:
        x, y = pts[0], pts[1]
        patches_labels[y, x] = 1
    
    contours, hierarchy = cv2.findContours((patches_labels > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    convex_hulls = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > area_thresh:
            hull = cv2.convexHull(contour)
            filtered_contours.append(contour)
            convex_hulls.append(hull)
    
    return filtered_contours, convex_hulls, patches_labels, patches_labels.shape[0], patches_labels.shape[1]

def is_point_on_line_segment(start, end, point, tol=1e-9):
    (x1, y1), (x2, y2), (x, y) = start, end, point
    cross_product = (y - y1) * (x2 - x1) - (x - x1) * (y2 - y1)
    
    if abs(cross_product) > tol:
        return False
    
    dot_product = (x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)
    if dot_product < 0:
        return False
    
    squared_length = (x2 - x1) ** 2 + (y2 - y1) ** 2
    if dot_product > squared_length:
        return False
    
    return True

def Point_in_Region(point, polygon):
    poly_path = mpath.Path(polygon)
    is_inside = poly_path.contains_point(point)
    if is_inside:
        return True
    
    for i in range(len(polygon)):
        start_point = polygon[i]
        end_point = polygon[(i + 1) % len(polygon)]
        if is_point_on_line_segment(start_point, end_point, point):
            return True
    
    return False

def zero_tolerance(float_num):
    if abs(float_num) <= 1e-3:
        return 0
    else:
        return float_num

def ND_zscore_filter(contour_df, weight):
    scaler = StandardScaler()
    contour_df_copy = contour_df.copy()

    # Compute z-scores for 'num' and 'density' columns
    zscores = scaler.fit_transform(contour_df_copy[['number', 'density']])
    contour_df_copy['number_zscore'] = zscores[:, 0]
    contour_df_copy['density_zscore'] = zscores[:, 1]

    # Sum the z-scores with weights
    contour_df_copy['zscore_sum'] = weight[0] * contour_df_copy['number_zscore'] + weight[1] * contour_df_copy['density_zscore']

    # Apply zero tolerance
    contour_df_copy['zscore_sum'] = contour_df_copy['zscore_sum'].apply(zero_tolerance)

    return contour_df_copy

def find_contour(wsi, sorted_all_pts, state, cl, area_thresh, all_patches, selected_patches):
    patches_in_hulls = []
    tp_in_regions, fp_in_regions = {}, {}
    patches_in_regions = {}

    ### Get contours ###
    contours, hulls, _, _, _ = find_contours_of_connected_components(label=cl, sorted_pts=sorted_all_pts, area_thresh=area_thresh)
    regions = contours_processing(contours, forImage=False)
    regions_hulls = contours_processing(hulls, forImage=False)

    ### Select data in regions ###
    for pts in tqdm(sorted_all_pts):
        ptx, pty, pseudo_label = pts[0], pts[1], pts[2]
        left_up = [int(ptx), int(pty)]
        # right_up = [int(ptx) + 1, int(pty)]
        # left_down = [int(ptx), int(pty) + 1]
        # right_down = [int(ptx) + 1, int(pty) + 1]

        cl_text = "HCC" if cl == "H" else "Normal"
        formatted_filename = (
            f'C{wsi}_{cl_text}-{int(ptx * pts_ratio):05d}-{int(pty * pts_ratio):05d}-{pts_ratio:05d}x{pts_ratio:05d}.tif'
            if state == "old"
            else f'{int(ptx * pts_ratio)}_{int(pty * pts_ratio)}.tif'
        )

        ### Check pts in every HCC region or not ###
        true_label_index = {'N': 0, 'H': 1, 'C': 2}[cl]
        for idx, region in enumerate(regions):
            if ((Point_in_Region(left_up, region)==True)):
                if (formatted_filename in all_patches) and (formatted_filename not in selected_patches['file_name']):
                    selected_patches['file_name'].append(formatted_filename)
                    selected_patches['label'].append(cl)

                    if idx not in patches_in_regions.keys():
                        patches_in_regions[idx] = []
                    patches_in_regions[idx].append(formatted_filename)

                    if idx not in tp_in_regions.keys():
                        tp_in_regions[idx] = 0
                    if idx not in fp_in_regions.keys():
                        fp_in_regions[idx] = 0
                    
                    if pseudo_label == true_label_index:
                        tp_in_regions[idx] += 1
                    else:
                        fp_in_regions[idx] += 1
        
        for region_hull in regions_hulls:
            if ((Point_in_Region(left_up, region_hull)==True)):
                if formatted_filename not in patches_in_hulls:
                    patches_in_hulls.append(formatted_filename)

    return selected_patches, patches_in_regions, tp_in_regions, fp_in_regions

def zscore_filter(tp_in_cancer_regions, fp_in_cancer_regions, tn_in_norm_regions, fn_in_norm_regions, patches_in_cancer_regions, patches_in_norm_regions, cl):
    pos_num, neg_num = 0,0
    positive_cases = {"contour_key": [], "number": [], "density": []}
    for key in tp_in_cancer_regions.keys():
        num = tp_in_cancer_regions[key] + fp_in_cancer_regions[key]
        den = tp_in_cancer_regions[key] / (tp_in_cancer_regions[key] + fp_in_cancer_regions[key])
        positive_cases["contour_key"].append(key)
        positive_cases["number"].append(num)
        positive_cases["density"].append(den)
        pos_num += num
    pos_df = pd.DataFrame(positive_cases)

    negative_cases = {"contour_key": [], "number": [], "density": []}
    for key in tn_in_norm_regions.keys():
        num = tn_in_norm_regions[key] + fn_in_norm_regions[key]
        den = tn_in_norm_regions[key] / (tn_in_norm_regions[key] + fn_in_norm_regions[key])
        negative_cases["contour_key"].append(key)
        negative_cases["number"].append(num)
        negative_cases["density"].append(den)
        neg_num += num
    neg_df = pd.DataFrame(negative_cases)

    # num_filter_pos_df = pos_df[pos_df["number"] >= num_thresh]
    # num_filter_neg_df = neg_df[neg_df["number"] >= num_thresh]

    ideal_patches = {'file_name': [], 'label': []}
    
    if pos_num >0:
        pl_cancer_contour_df = ND_zscore_filter(contour_df=pos_df, weight=[1, 1])  # z-score
        # pl_cancer_contour_df.to_csv(f"{save_path}/Data/{_wsi}_Gen{gen}_tpfp_ND_zscore_filtered_contour_by_Gen{gen-1}.csv")
        # Filter keys where the sum of z-scores is greater than or equal to 0
        pl_cancer_filtered_keys = pl_cancer_contour_df[pl_cancer_contour_df['zscore_sum'] >= 0]['contour_key'].to_list()
        for pl_cancer_key in pl_cancer_filtered_keys:
            ideal_patches['file_name'].extend(patches_in_cancer_regions[pl_cancer_key])
            ideal_patches['label'].extend([cl] * len(patches_in_cancer_regions[pl_cancer_key]))

    else:
        pl_cancer_contour_df = pos_df
        # pl_cancer_contour_df.to_csv(f"{save_path}/Data/{_wsi}_Gen{gen}_tpfp_ND_zscore_filtered_contour_by_Gen{gen-1}.csv")
        print('NO cancer')

    if neg_num >0:
        pl_norm_contour_df = ND_zscore_filter(contour_df=neg_df, weight=[1, 1])
        # pl_norm_contour_df.to_csv(f"{save_path}/Data/{_wsi}_Gen{gen}_tnfn_ND_zscore_filtered_contour_by_Gen{gen-1}.csv")
        pl_norm_filtered_keys = pl_norm_contour_df[pl_norm_contour_df['zscore_sum'] >= 0]['contour_key'].to_list()
        for pl_norm_key in pl_norm_filtered_keys:
            ideal_patches['file_name'].extend(patches_in_norm_regions[pl_norm_key])
            ideal_patches['label'].extend(['N'] * len(patches_in_norm_regions[pl_norm_key]))

    else:
        pl_norm_contour_df=neg_df
        # pl_norm_contour_df.to_csv(f"{save_path}/Data/{_wsi}_Gen{gen}_tnfn_ND_zscore_filtered_contour_by_Gen{gen-1}.csv")
        print('NO Normal')

    return ideal_patches, pl_cancer_contour_df, pl_norm_contour_df

def zscore_filter_multi_class(tp_in_regions, fp_in_regions, patches_in_regions, cl):
    pos_num = 0
    positive_cases = {"contour_key": [], "number": [], "density": []}
    for key in tp_in_regions.keys():
        num = tp_in_regions[key] + fp_in_regions[key]
        den = tp_in_regions[key] / (tp_in_regions[key] + fp_in_regions[key])
        positive_cases["contour_key"].append(key)
        positive_cases["number"].append(num)
        positive_cases["density"].append(den)
        pos_num += num
    pos_df = pd.DataFrame(positive_cases)

    # num_filter_pos_df = pos_df[pos_df["number"] >= num_thresh]
    # num_filter_neg_df = neg_df[neg_df["number"] >= num_thresh]

    ideal_patches = {'file_name': [], 'label': []}
    
    if pos_num >0:
        pl_contour_df = ND_zscore_filter(contour_df=pos_df, weight=[1, 1])  # z-score
        # Filter keys where the sum of z-scores is greater than or equal to 0
        pl_filtered_keys = pl_contour_df[pl_contour_df['zscore_sum'] >= 0]['contour_key'].to_list()
        for pl_key in pl_filtered_keys:
            ideal_patches['file_name'].extend(patches_in_regions[pl_key])
            ideal_patches['label'].extend([cl] * len(patches_in_regions[pl_key]))

    else:
        pl_contour_df = pos_df
        print(f'NO {cl}')

    return ideal_patches, pl_contour_df