import cv2 as cv
import numpy as np

CORNER_SIZE = 45
SIGNS = [[1, 1], [1, -1], [-1, -1], [-1, 1]]


def remove_corner(img: np.ndarray, color=(0, 0, 0), size=CORNER_SIZE) -> np.ndarray:
    H, W = img.shape[:2]
    corners = [[0, 0], [0, H], [W, H], [W, 0]]
    shape = np.array([[0, 0], [0, size], [size, 0]])
    for corner, sign in zip(corners, SIGNS):
        cur = corner + shape * sign
        cv.fillPoly(img, [cur.astype(np.int32)], color)
    return img
