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
    
    if label == 'H':
        cate_pts = sorted_pts[sorted_pts[:, 2] == 0]
    if label == 'N':
        cate_pts = sorted_pts[sorted_pts[:, 2] == 1]
    if label == 'F':
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

def find_contours(wsi, sorted_all_pts, state, cl, area_thresh, all_patches, selected_patches):
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
        for idx, region in enumerate(regions):
            if ((Point_in_Region(left_up, region)==True)):
                if (formatted_filename in all_patches) and (formatted_filename not in selected_patches['filename']):
                    selected_patches['filename'].append(formatted_filename)
                    selected_patches['label'].append(cl)

                    if idx not in patches_in_regions.keys():
                        patches_in_regions[idx] = []
                    patches_in_regions[idx].append(formatted_filename)

                    if idx not in tp_in_regions.keys():
                        tp_in_regions[idx] = 0
                    if idx not in fp_in_regions.keys():
                        fp_in_regions[idx] = 0
                    
                    if pseudo_label == 0:
                        tp_in_regions[idx] += 1
                    if pseudo_label == 1:
                        fp_in_regions[idx] += 1
        
        for region_hull in regions_hulls:
            if ((Point_in_Region(left_up, region_hull)==True)):
                if formatted_filename not in patches_in_hulls:
                    patches_in_hulls.append(formatted_filename)

        return selected_patches, patches_in_regions, tp_in_regions, fp_in_regions