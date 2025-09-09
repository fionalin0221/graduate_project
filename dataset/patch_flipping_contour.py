import os
import numpy as np
import cv2
import matplotlib.path as mpath
from sklearn.preprocessing import StandardScaler
import pandas as pd

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

def find_areas(binary_img):
    return cv2.findContours((binary_img > 0).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    
def flip_patch(binary_img, contours, hierarchy, area_thresh):
    
    def get_contour_depth(hierarchy, idx):
        depth = 0
        parent = hierarchy[0][idx][3]
        while parent != -1:
            depth += 1
            parent = hierarchy[0][parent][3]
        return depth

    filtered_idx = []
    filtered_contours = []
    contour_depths = []
    areas = []

    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > area_thresh:
            filtered_idx.append(idx)
            filtered_contours.append(contour)
            depth = get_contour_depth(hierarchy, idx)
            contour_depths.append(depth)
            areas.append(area)
            
    filtered_hierarchy = []
    index_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(filtered_idx)}
    for idx in filtered_idx:
        next_idx, prev_idx, child_idx, parent_idx = hierarchy[0][idx]
        new_next   = index_map.get(next_idx, -1)
        new_prev   = index_map.get(prev_idx, -1)
        new_child  = index_map.get(child_idx, -1)
        new_parent = index_map.get(parent_idx, -1)

        filtered_hierarchy.append([new_next, new_prev, new_child, new_parent])

    parent_to_children = {}
    
    for i in range(len(filtered_hierarchy)):
        parent = filtered_hierarchy[i][3]  # Parent index
        if parent != -1:
            if parent not in parent_to_children:
                parent_to_children[parent] = []
            parent_to_children[parent].append(i)

    regions = []
    for contour in filtered_contours:
        contour = contour.reshape(-1, 2)
        regions.append(contour)

    h, w = binary_img.shape
    # output_img = np.zeros((h, w))
    output_img = np.copy(binary_img)

    for pty in range(h):
        for ptx in range(w):
            left_up = [int(ptx), int(pty)]
            # right_up = [int(ptx) + 1, int(pty)]
            # left_down = [int(ptx), int(pty) + 1]
            # right_down = [int(ptx) + 1, int(pty) + 1]

            for idx, region in enumerate(regions):
                # print(areas[idx], contour_depths[idx])
                if contour_depths[idx] == 0:
                    if  Point_in_Region(left_up, region):
                        holes = parent_to_children.get(idx, [])

                        inside_hole = False
                        for hole in holes:
                            hole_region = regions[hole]
                            if Point_in_Region(left_up, hole_region):
                                inside_hole = True
                                break
                        
                        if not inside_hole:
                            output_img[pty, ptx] = 1
    return output_img


if __name__ == '__main__':
    # img = np.array(
    #     [[0, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 1, 1, 0, 0, 0, 1, 0],
    #      [0, 1, 0, 1, 0, 0, 1, 0],
    #      [0, 1, 1, 1, 0, 1, 1, 1],
    #      [1, 1, 1, 0, 0, 1, 0, 1],
    #      [1, 0, 1, 0, 0, 1, 0, 1],
    #      [0, 1, 0, 1, 0, 1, 0, 1],
    #      [0, 0, 1, 1, 0, 1, 1, 1]
    #     ], dtype=np.uint8
    # )
    pts_ratio = 448

    df = pd.read_csv(f"/workspace/Data/Results/Mix_NDPI/Generation_Training/100WTC_LP_3200/10138/trial_5/TI/10138_3_class_all_patches_filter_v2_TI.csv")
    all_patches = df['file_name'].to_list()
    selected_columns = []
    for cl in ["N", "H", "C"]:
        selected_columns.append(f"{cl}_pred")
    selected_data = df[selected_columns].to_numpy()

    ### Get (x, y, pseudo-label) of every patch ###
    all_pts = []
    for idx, img_name in enumerate(all_patches):
        x, y = img_name[:-4].split('_')
        row = selected_data[idx, :]
        max_col = np.argmax(row)
        all_pts.append([(int(x)) // pts_ratio, (int(y)) // pts_ratio, max_col])
    all_pts = np.array(all_pts)

    ### First sorted pts on x, then on y ###
    sorted_index = np.lexsort((all_pts[:, 1], all_pts[:, 0]))
    sorted_all_pts = all_pts[sorted_index] # x, y, label

    selected_patches = {'file_name': [], 'label': []}

    for idx, cl in enumerate(["N"]):
        print(f"running for {cl} ...")

        ### Make HCC or Normal components regions ###
        x_max = np.max(sorted_all_pts[:, 0])
        y_max = np.max(sorted_all_pts[:, 1])
        binary_img = np.zeros((y_max+1, x_max+1), np.uint8)  # for connected-components of HCC/Normal patches
        h, w = binary_img.shape

        cate_pts = sorted_all_pts[sorted_all_pts[:, 2] == idx]

        for pts in cate_pts:
            x, y = pts[0], pts[1]
            binary_img[y, x] = 1

        img = binary_img
        h, w = img.shape

        contours, hierarchy = find_areas(img)
        img_filtered = flip_patch(img, contours, hierarchy, area_thresh=3)

        # print("Filtered image:\n", img_filtered)
        save_img = (img_filtered * 255).astype(np.uint8)
        cv2.imwrite("contour_filtered_result.png", save_img)