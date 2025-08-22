import numpy as np

# 8-neighborhood offsets (Freeman chain code directions 0..7)
DELTAS = [
    ( 1,  0),  # 0: right
    ( 1, -1),  # 1: top-right
    ( 0, -1),  # 2: top
    (-1, -1),  # 3: top-left
    (-1,  0),  # 4: left
    (-1,  1),  # 5: bottom-left
    ( 0,  1),  # 6: bottom
    ( 1,  1),  # 7: bottom-right
]

def fetch_contour(img, x0, y0, method):
    '''
    Trace a single contour starting at (x0, y0).
    '''
    h, w = img.shape
    assert 0 <= x0 < w and 0 <= y0 < h
    assert img[y0, x0] == 1  # must be a foreground border pixel
    assert method in ("NONE", "SIMPLE"), f"Invalid method: {method}, must be 'NONE' or 'SIMPLE'"

    all_contour = []
    contour = []

    i0 = (x0, y0)
    # print(f"first point {i0}")

    # initial search direction
    s = 4 
    found = False
    for _ in range(8):
        s = (s-1) % 8
        dx, dy = DELTAS[s]
        nx, ny = i0[0] + dx, i0[1] + dy
        if 0 <= nx < w and 0 <= ny < h and img[ny, nx] != 0:
            found = True
            break
    if not found:
        return [(x, y)]
    
    i1 = (x0+dx, y0+dy)
    # print(f"last point {i1}")
    i3 = i0
    prev_s = (s+4) % 8

    while True:
        ns = s
        for _ in range(8):
            ns = (ns+1) % 8
            nx, ny = i3[0] + DELTAS[ns][0], i3[1] + DELTAS[ns][1]
            if 0 <= nx < w and 0 <= ny < h and img[ny, nx] != 0:
                s = ns
                i4= (nx, ny)
                break
        
        all_contour.append(i3)
        if method == "NONE":
            contour.append(i3)
        elif method == "SIMPLE":
            if s != prev_s:
                contour.append(i3)
                prev_s = s

        if i4 == i0 and i3 == i1:
            break

        s = (s+4) % 8
        i3 = i4

    all_contour.append(i1)
    if method == "NONE":
        contour.append(i1)
    elif method == "SIMPLE":
        if s != prev_s:
            contour.append(i1)
    return contour, all_contour


def cv2_find_contours(img, method):
    '''
    img: 2D numpy array (0=background, 1=foreground)
    method: SIMPLE/NONE
    *** only consider EXTERNAL contours ***
    '''
    h, w = img.shape
    img = (img > 0).astype(np.uint8)
    contours = []

    for y in range(h):
        for x in range(w):
            if img[y, x] == 1:    # check if this is a border pixel
                neighbors = [
                    img[ny, nx] if 0 <= nx < w and 0 <= ny < h else 0
                    for dx, dy in DELTAS
                    for nx, ny in [(x + dx, y + dy)]
                ]
                if any(n == 0 for n in neighbors):  # has background neighbor
                    contour, all_contour = fetch_contour(img, x, y, method)
                    contours.append(contour)
                    for (cx, cy) in all_contour:
                        img[cy, cx] = 2  # mark as visited

    return contours


# Example test
if __name__ == '__main__':
    # img = np.zeros((10, 10), dtype=np.uint8)
    # img[3:7, 3:7] = 1   # simple 4x4 square

    img = np.zeros((10, 12), dtype=np.uint8)

    # L-shape
    img[1:5, 1:3] = 1
    img[4, 1:6] = 1

    # Small square
    img[6:9, 7:10] = 1
    img[7, 8] = 0

    # Thin vertical line
    img[0:5, 10] = 1

    print(img)
    
    contours = cv2_find_contours(img, method='SIMPLE')
    for c in contours:
        print("Contour points:", c)