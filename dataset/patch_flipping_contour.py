import os
import numpy as np
import cv2
import matplotlib.path as mpath
from sklearn.preprocessing import StandardScaler

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
    return cv2.findContours((binary_img > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
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

    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > area_thresh:
            filtered_idx.append(idx)
            filtered_contours.append(contour)
            depth = get_contour_depth(hierarchy, idx)
            contour_depths.append(depth)
            
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
    output_img = np.zeros((h, w))

    for pty in range(h):
        for ptx in range(w):
            left_up = [int(ptx), int(pty)]
            # right_up = [int(ptx) + 1, int(pty)]
            # left_down = [int(ptx), int(pty) + 1]
            # right_down = [int(ptx) + 1, int(pty) + 1]

            for idx, region in enumerate(regions):
                if contour_depths[idx] % 2 == 0:
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
    img = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 1, 0, 0, 0, 1, 0],
         [0, 1, 0, 1, 0, 0, 1, 0],
         [0, 1, 1, 1, 0, 1, 1, 1],
         [1, 1, 1, 0, 0, 1, 0, 1],
         [1, 0, 1, 0, 0, 1, 0, 1],
         [0, 1, 0, 1, 0, 1, 0, 1],
         [0, 0, 1, 1, 0, 1, 1, 1]
        ], dtype=np.uint8
    )

    h, w = img.shape

    contours, hierarchy = find_areas(img)
    img_filtered = flip_patch(img, contours, hierarchy, area_thresh=3)

    print("Filtered image:\n", img_filtered)