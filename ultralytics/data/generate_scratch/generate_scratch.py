from collections.abc import Generator

import cv2 as cv
import numpy as np

from .remove_corner import CORNER_SIZE, remove_corner

LENGTH_RANGE = (10, 100)  # maximum distance between two end points
END_BRUSH_RANGE = (0, 1)  # brush size range of the two end points
MID_BRUSH_RANGE = (2, 5)  # brush size range of the mid point
END_VARIANT_RANGE = 5
MID_VARIANT_RANGE = (10, np.deg2rad(5))
CLUSTER_CNT_RANGE = (2, 5)
SCRATCH_CNT = 30


def bezier(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Generator[np.ndarray, None, None]:
    def calc(t):
        return t * t * p1 + 2 * t * (1 - t) * p2 + (1 - t) * (1 - t) * p3

    # get the approximate pixel count of the curve
    approx = cv.arcLength(np.array([calc(t)[:2] for t in np.linspace(0, 1, 10)], dtype=np.float32), False)
    for t in np.linspace(0, 1, round(approx * 2)):
        yield np.round(calc(t)).astype(np.int32)


def generate_scratch_fixed_end(
    img: np.ndarray,
    color,
    p1: np.ndarray,
    p3: np.ndarray,
    mid_brush_range: tuple[float, float] = MID_BRUSH_RANGE,
) -> np.ndarray:
    rho1 = np.linalg.norm(p1[:2] - p3[:2])

    # generate the middle point
    rho2, theta2 = np.random.uniform([0], [rho1 / 2, np.pi * 2])
    p2 = (p1 + p3) / 2 + [rho2 * np.cos(theta2), rho2 * np.sin(theta2), 0]

    # generate the brush sizes of the 3 points
    p2[2] = np.random.uniform(mid_brush_range[0], mid_brush_range[1])

    for x, y, brush in bezier(p1, p2, p3):
        cv.circle(img, (x, y), brush, color, -1)
    return img


def generate_scratch(
    img: np.ndarray,
    color,
    length_range: tuple[float] = LENGTH_RANGE,
    end_brush_range: tuple[float, float] = END_BRUSH_RANGE,
    mid_brush_range: tuple[float, float] = MID_BRUSH_RANGE,
) -> np.ndarray:
    H, W = img.shape
    # generate the 2 end points of the bezier curve
    x, y, rho1, theta1 = np.random.uniform([0, 0, length_range[0], 0], [W, H, length_range[1], np.pi * 2])
    p1 = np.array([x, y, 0])
    p3 = p1 + [rho1 * np.cos(theta1), rho1 * np.sin(theta1), 0]

    # generate the middle point
    rho2, theta2 = np.random.uniform([0], [rho1 / 2, np.pi * 2])
    p2 = (p1 + p3) / 2 + [rho2 * np.cos(theta2), rho2 * np.sin(theta2), 0]

    # generate the brush sizes of the 3 points
    p1[2], p2[2], p3[2] = np.random.uniform(*np.transpose([end_brush_range, mid_brush_range, end_brush_range]))

    for x, y, brush in bezier(p1, p2, p3):
        cv.circle(img, (x, y), brush, color, -1)
    return img


def generate_scratch_cluster(
    img: np.ndarray,
    color,
    length_range: tuple[float, float] = LENGTH_RANGE,
    end_brush_range: tuple[float, float] = END_BRUSH_RANGE,
    mid_brush_range: tuple[float, float] = MID_BRUSH_RANGE,
    end_variant_range: float = END_VARIANT_RANGE,
    mid_variant_range: float = MID_VARIANT_RANGE,
    cluster_cnt_range: tuple[int, int] = CLUSTER_CNT_RANGE,
    remove_corner_size: float = CORNER_SIZE,
) -> np.ndarray:
    if (cnt := np.random.randint(*cluster_cnt_range)) <= 0:
        return img
    H, W = img.shape
    # generate the 2 end points of the bezier curve
    x, y, rho1, theta1 = np.random.uniform([0, 0, length_range[0], 0], [W, H, length_range[1], np.pi * 2])
    p1_ = np.array([x, y, 0])
    p3_ = p1_ + [rho1 * np.cos(theta1), rho1 * np.sin(theta1), 0]
    rho2_, theta2_ = np.random.uniform([0], [rho1 / 2, np.pi * 2])

    cur_scratch_mask = np.zeros((H, W), np.uint8)

    contours = []
    for _ in range(cnt - 1):
        p1, p3 = p1_.copy(), p3_.copy()
        end_variant = np.random.uniform(-end_variant_range, end_variant_range, size=4)
        p1[:2] += end_variant[:2]
        p3[:2] += end_variant[2:]
        mid_variant = np.random.uniform([max(-rho2_, -mid_variant_range[0]), -mid_variant_range[1]], mid_variant_range)
        rho2, theta2 = [rho2_, theta2_] + mid_variant
        p2 = (p1 + p3) / 2 + [rho2 * np.cos(theta2), rho2 * np.sin(theta2), 0]
        p1[2], p2[2], p3[2] = np.random.uniform(*np.transpose([end_brush_range, mid_brush_range, end_brush_range]))
        p2[2] = max(1.0, p2[2])

        curve = np.array(list(bezier(p1, p2, p3)))
        for x, y, brush in curve:
            if not (0 <= x < W and 0 <= y < H):
                break
            cv.circle(cur_scratch_mask, (x, y), brush + 1, 255, -1)

        PADDING = max(end_brush_range[1], mid_brush_range[1]) + 2
        minx, miny = curve[:, :2].min(axis=0).astype(np.int32) - PADDING
        maxx, maxy = curve[:, :2].max(axis=0).astype(np.int32) + PADDING
        minx, miny = max(minx, 0), max(miny, 0)
        maxx, maxy = min(maxx, W - 1), min(maxy, H - 1)
        remove_corner(cur_scratch_mask, 0, remove_corner_size)
        contour_region = cur_scratch_mask[miny : maxy + 1, minx : maxx + 1]
        if not (cur_contours := cv.findContours(contour_region, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]):
            continue

        contour = max(cur_contours, key=cv.contourArea)
        contour_length = cv.arcLength(contour, True) / 2
        if contour_length < length_range[0]:
            continue

        contours.append((contour + [minx, miny]).reshape((-1, 2)))

        for x, y, brush in curve:
            if not (0 <= x < W and 0 <= y < H):
                break
            cv.circle(img, (x, y), brush, color, -1)
            cv.circle(cur_scratch_mask, (x, y), brush + 1, 0, -1)

    return img, contours


if __name__ == "__main__":
    W, H = 640, 480

    img = np.zeros((H, W), np.uint8)
    for _ in range(SCRATCH_CNT):
        generate_scratch(img, 255)

    cv.imshow("img", img)
    cv.waitKey(0)
