import cv2 as cv
import numpy as np

from .generate_perlin_noise_2d import generate_perlin_noise_2d
from .generate_scratch import generate_scratch_cluster
from .remove_corner import remove_corner


def apply_scratch(img: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    H, W = img.shape[:2]

    mask_noise = generate_perlin_noise_2d((H, W), (32, 32))
    mask_noise = cv.normalize(mask_noise, None, 0.5, 0.75, cv.NORM_MINMAX)

    dark_img = cv.blur(img, (5, 5))
    dark_img = cv.cvtColor(dark_img, cv.COLOR_BGR2HSV)
    dark_img[..., 2] = np.round(dark_img[..., 2] * mask_noise).astype(np.uint8)
    dark_img = cv.cvtColor(dark_img, cv.COLOR_HSV2BGR)

    mask = np.zeros((H, W), np.uint8)
    contours = []
    for _ in range(np.random.randint(1, 20)):
        mask, cur_contours = generate_scratch_cluster(
            mask, 255, (50, 200), (0, 1), (1, 3), 10, (40, np.deg2rad(10)), (1, 9)
        )
        contours.extend(cur_contours)

    blur_mask = cv.blur(mask, (5, 5))
    final_mask = (blur_mask.astype(np.float32) / 255)[..., np.newaxis]

    final_mask = remove_corner(final_mask, 0)

    result_img = np.round(img * (1 - final_mask) + dark_img * final_mask).astype(np.uint8)

    if np.random.rand() < 0.8:
        noise = generate_perlin_noise_2d((H, W), (8, 8))
        noise = cv.normalize(noise, None, 0.9, 1, cv.NORM_MINMAX)
        result_img = cv.cvtColor(result_img, cv.COLOR_BGR2HSV)
        result_img[..., 2] = np.clip(np.round(result_img[..., 2] * noise), 0, 255).astype(np.uint8)
        result_img = cv.cvtColor(result_img, cv.COLOR_HSV2BGR)

    return result_img, contours


if __name__ == "__main__":
    IMG = "datasets/bad.jpg"
    img = cv.imread(IMG)
    result_img, contours = apply_scratch(img)

    result_img_annotated = result_img.copy()
    for contour in contours:
        cv.polylines(result_img_annotated, [contour], True, np.random.randint(0, 255, (3,)).tolist(), 1)

    cv.imshow("img", img)
    cv.imshow("result_img", result_img)
    cv.imshow("result_img_annotated", result_img_annotated)
    cv.waitKey(0)
