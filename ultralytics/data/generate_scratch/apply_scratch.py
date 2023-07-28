import cv2 as cv
import numpy as np

from .generate_perlin_noise_2d import generate_perlin_noise_2d
from .generate_scratch import generate_scratch, generate_scratch_cluster
from .remove_corner import remove_corner


def apply_scratch(img: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    H, W = img.shape[:2]

    mask_noise = generate_perlin_noise_2d((H, W), (32, 32))

    dark_img1 = cv.blur(img, (5, 5))
    dark_img1 = cv.cvtColor(dark_img1, cv.COLOR_BGR2HSV)
    dark_img2 = dark_img1.copy()
    dark_img1[..., 2] = np.round(dark_img1[..., 2] * cv.normalize(mask_noise, None, 0.3, 0.5, cv.NORM_MINMAX)).astype(np.uint8)
    dark_img2[..., 2] = np.round(dark_img2[..., 2] * cv.normalize(mask_noise, None, 0.6, 0.8, cv.NORM_MINMAX)).astype(np.uint8)
    dark_img1 = cv.cvtColor(dark_img1, cv.COLOR_HSV2BGR)
    dark_img2 = cv.cvtColor(dark_img2, cv.COLOR_HSV2BGR)

    mask1 = np.zeros((H, W), np.uint8)
    mask2 = np.zeros((H, W), np.uint8)

    for _ in range(np.random.randint(1, 10)):
        mask1 = generate_scratch(mask1, 255, (3, 5), (1, 2), (1, 4))
    mask1 = remove_corner(mask1, 0)
    contours = [c.reshape(-1, 2) for c in cv.findContours(mask1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]]
    mask1 = cv.blur(mask1, (3, 3))
    mask1 = mask1[..., np.newaxis] / 255

    for _ in range(np.random.randint(1, 20)):
        mask2, cur_contours = generate_scratch_cluster(mask2, 255, (50, 200), (0, 1), (1, 3), 10, (40, np.deg2rad(10)), (1, 9))
        contours.extend(cur_contours)
    mask2 = cv.GaussianBlur(mask2, (5, 5), 2)
    mask2 = remove_corner(mask2, 0)
    mask2 = mask2[..., np.newaxis] / 255

    result_img = img * (1 - mask1) + dark_img1 * mask1
    result_img = result_img * (1 - mask2) + dark_img2 * mask2
    result_img = np.round(result_img).astype(np.uint8)

    if np.random.rand() < 0.8:
        noise = generate_perlin_noise_2d((H, W), (8, 8))
        noise = cv.normalize(noise, None, 0.9, 1, cv.NORM_MINMAX)
        result_img = cv.cvtColor(result_img, cv.COLOR_BGR2HSV)
        result_img[..., 2] = np.clip(np.round(result_img[..., 2] * noise), 0, 255).astype(np.uint8)
        result_img = cv.cvtColor(result_img, cv.COLOR_HSV2BGR)

    return result_img, contours
