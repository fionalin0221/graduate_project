import numpy as np
import cv2

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

    img_pad = np.zeros((h+2, w+2))
    img_pad[1:1+h, 1:1+w] = img

    num_labels, labels, stats = find_areas(img_pad)

    img_pad_filtered = flip_patch(img_pad, num_labels, labels, stats, area_thresh=3)
    img_filtered = img_pad_filtered[1:-1, 1:-1]

    print("Filtered image:\n", img_filtered)