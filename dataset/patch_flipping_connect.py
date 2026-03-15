import numpy as np
import cv2
import pandas as pd

def find_areas(img_pad):
    num_labels, graph, stats, _ = cv2.connectedComponentsWithStats(img_pad.astype(np.uint8), connectivity=4)
    return num_labels, graph, stats

def flip_patch(components, components_graph, original_img_pad, area_thresh):
    flip_pts_dict = {}

    while components:
        comp_id = min(components, key=lambda k: components[k][0])
        area, original_cl, stats = components[comp_id]

        if area > area_thresh:
            break
        
        x, y, w, h, _ = stats

        # Add 1-pixel padding safely
        y1 = max(0, y - 1)
        y2 = min(original_img_pad.shape[0], y + h + 1)
        x1 = max(0, x - 1)
        x2 = min(original_img_pad.shape[1], x + w + 1)

        # Crop the padded bounding box
        local_img = original_img_pad[y1:y2, x1:x2]
        local_graph = components_graph[y1:y2, x1:x2]
        bbox = (local_graph == comp_id).astype(np.uint8)

        # Dilate to find the neighbors of bbox
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(bbox, kernel, iterations=1)
        neighbor_mask = (dilated - bbox) == 1 # 2D bollean

        neighbor_labels = local_img[neighbor_mask] # 1D array
        neighbor_ids = local_graph[neighbor_mask] # 1D array

        valid_mask = (neighbor_ids != 0) & (neighbor_ids != comp_id)
        valid_neighbor_labels = neighbor_labels[valid_mask]
        valid_neighbor_ids = neighbor_ids[valid_mask]

        target_cl = None
        target_comp_id = None

        if len(valid_neighbor_labels) > 0:
            unique_cls, counts_cls = np.unique(valid_neighbor_labels, return_counts=True)
            target_cl = unique_cls[np.argmax(counts_cls)]

            valid_comp_ids = valid_neighbor_ids[valid_neighbor_labels == target_cl]
            unique_ids, counts_ids = np.unique(valid_comp_ids, return_counts=True)
            target_comp_id = unique_ids[np.argmax(counts_ids)]

        if target_comp_id is not None:
            ys, xs = np.where(components_graph == comp_id)
            for _x, _y in zip(xs, ys):
                flip_pts_dict[_x-1, _y-1] = int(target_cl)

                components_graph[_y, _x] = target_comp_id
                original_img_pad[_y, _x] = target_cl

            if target_comp_id in components:
                target_ys, target_xs = np.where(components_graph == target_comp_id)
                tx, ty, tw, th = np.min(target_xs), np.min(target_ys), np.max(target_xs)-np.min(target_xs)+1, np.max(target_ys)-np.min(target_ys)+1

                ta = components[target_comp_id][0] + area
                components[target_comp_id][0] = ta
                components[target_comp_id][2] = [tx, ty, tw, th, ta]
        
        components.pop(comp_id)

    return [[pos[0], pos[1], cl] for pos, cl in flip_pts_dict.items()]


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
        # max_col = np.argmax(row)
        mask = row > 0.5                # boolean array
        if mask.sum() == 1:             # exactly one True
            cl_idx = mask.argmax()  
            all_pts.append([(int(x)) // pts_ratio, (int(y)) // pts_ratio, cl_idx])
    all_pts = np.array(all_pts)

    ### First sorted pts on x, then on y ###
    sorted_index = np.lexsort((all_pts[:, 1], all_pts[:, 0]))
    sorted_all_pts = all_pts[sorted_index] # x, y, label

    # convert to 1-based labels: N=1, H=2, C=3
    labels_1based = sorted_all_pts[:, 2] + 1

    # get max coordinates to define image size
    x_max = np.max(sorted_all_pts[:, 0])
    y_max = np.max(sorted_all_pts[:, 1])

    # create empty image (background=0)
    original_image = np.zeros((y_max + 1, x_max + 1), dtype=np.uint8)

    # fill image
    for (x, y), label in zip(sorted_all_pts[:, :2], labels_1based):
        original_image[y, x] = label
    
    # original_image: 2D array with values 0 (background), 1 (N), 2 (H), 3 (C)
    h, w = original_image.shape

    # create an RGB image
    color_image = np.zeros((h, w, 3), dtype=np.uint8)

    # assign colors
    color_map = {
        1: (0, 255, 0),   # green
        2: (0, 0, 255),   # red
        3: (255, 0, 0)    # blue
    }

    for label, color in color_map.items():
        color_image[original_image == label] = color

    # save the image
    # cv2.imwrite("label_map.png", color_image)
    
    selected_patches = {'file_name': [], 'label': []}

    for idx, cl in enumerate(["H"]):
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

        # save_img = (binary_img * 255).astype(np.uint8)
        # cv2.imwrite("origin.png", save_img)

        img = binary_img
        h, w = img.shape

        img_pad = np.zeros((h+2, w+2))
        img_pad[1:1+h, 1:1+w] = img
        original_img_pad = np.zeros((h+2, w+2))
        original_img_pad[1:1+h, 1:1+w] = original_image

        num_labels, labels, stats = find_areas(img_pad)

        flip_pts = flip_patch(img_pad, original_img_pad, ["N", "H", "C"], num_labels, labels, stats, area_thresh=3)
        # print(flip_pts)
        # img_filtered = img_pad_filtered[1:-1, 1:-1]

        # # print("Filtered image:\n", img_filtered)
        # save_img = (img_filtered * 255).astype(np.uint8)
        # cv2.imwrite("connect_filtered_result.png", save_img)