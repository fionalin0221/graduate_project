import numpy as np
import cv2
import pandas as pd

def find_areas(img_pad):
    img_inv = (img_pad == 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img_inv, connectivity=4)

    return num_labels, labels, stats

def flip_patch(img_pad, num_labels, labels, stats, area_thresh):
    img_pad_filtered = img_pad.copy()
    for comp_id in range(1, num_labels):  # skip background
        area = stats[comp_id, cv2.CC_STAT_AREA]
        if area <= 3:
            img_pad_filtered[labels == comp_id] = 1
            
    return img_pad_filtered


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

        save_img = (binary_img * 255).astype(np.uint8)
        cv2.imwrite("origin.png", save_img)

        img = binary_img
        h, w = img.shape

        img_pad = np.zeros((h+2, w+2))
        img_pad[1:1+h, 1:1+w] = img

        num_labels, labels, stats = find_areas(img_pad)

        img_pad_filtered = flip_patch(img_pad, num_labels, labels, stats, area_thresh=3)
        img_filtered = img_pad_filtered[1:-1, 1:-1]

        # print("Filtered image:\n", img_filtered)
        save_img = (img_filtered * 255).astype(np.uint8)
        cv2.imwrite("connect_filtered_result.png", save_img)