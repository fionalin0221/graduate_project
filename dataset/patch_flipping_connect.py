import numpy as np
import cv2
import pandas as pd

def find_areas(img_pad):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img_pad.astype(np.uint8), connectivity=4)
    return num_labels, labels, stats

def flip_patch(img_pad, original_img_pad, classes, num_labels, labels, stats, area_thresh):
    flip_pts = []
    img_pad_filtered = img_pad.copy()
    for comp_id in range(1, num_labels):  # skip background
        x, y, w, h, area = stats[comp_id]
        if area <= area_thresh:
            # Add 1-pixel padding safely
            y1 = max(0, y - 1)
            y2 = min(img_pad.shape[0], y + h)
            x1 = max(0, x - 1)
            x2 = min(img_pad.shape[1], x + w)

            # Crop the padded bounding box
            bbox = np.zeros((h+2, w+2), dtype=np.uint8)
            ys, xs = np.where(labels == comp_id)
            for _x, _y in zip(xs, ys):
                bbox[_y - y1, _x - x1] = 1

            kernel = np.ones((3,3), np.uint8)
            dilated = cv2.dilate(bbox, kernel, iterations=1)

            # outside boundary pixels only
            outside = dilated - bbox

            # outside is your binary array: 1 = outside pixel, 0 = else
            out_ys, out_xs = np.where(outside == 1)  # returns row indices (y) and column indices (x)

            # convert bbox-relative coordinates to original image coordinates
            orig_out_ys = out_ys + y1
            orig_out_xs = out_xs + x1

            # get the labels at those positions
            labels_at_positions = original_img_pad[orig_out_ys, orig_out_xs]
            # print(labels_at_positions)
            counts = np.zeros(len(classes)+1)

            for val in labels_at_positions:
                counts[int(val)] += 1
            cl = np.argmax(counts)

            for _x, _y in zip(xs, ys):
                flip_pts.append([_x-1, _y-1, cl])

    return flip_pts


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