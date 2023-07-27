import numpy as np
from ultralytics.utils.instance import Instances

from .apply_scratch import apply_scratch


def GenerateScratch(labels: dict) -> dict:
    """
    Affine images and targets.

    Args:
        labels (dict): a dict of `bboxes`, `segments`, `keypoints`.
    """
    img = labels["img"]
    labels.pop("instances", None)

    scratched_img, contours = apply_scratch(img)
    bboxes = []
    for contour in contours:
        (minx, miny), (maxx, maxy) = contour.min(axis=0), contour.max(axis=0)
        bboxes.append([minx, miny, maxx, maxy])

    new_instances = Instances(
        bboxes=np.array(bboxes, np.float32) if bboxes else np.zeros((0, 4), np.float32),
        segments=[c.astype(np.float32) for c in contours],
        keypoints=None,
        bbox_format="xyxy",
        normalized=False,
    )
    H, W = img.shape[:2]
    new_instances.normalize(W, H)
    new_instances.convert_bbox("xywh")

    labels["img"] = scratched_img
    labels["instances"] = new_instances
    labels["cls"] = np.zeros((len(new_instances), 1), np.float32)

    return labels


if __name__ == "__main__":
    import cv2 as cv

    IMG = "dataset/bad.jpg"
    img = cv.imread(IMG)
    label = {"img": img}
    label = GenerateScratch(label)
    print(label)

    cv.imshow("img", img)
    cv.imshow("scratched_img", label["img"])
    cv.waitKey(0)
