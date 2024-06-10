import numpy as np
from skimage.draw import polygon
import matplotlib.pyplot as plt

import sys
sys.path.append(".")

from utils.grasp_utils import angle_from_grasp_rectangle


def iou(grasp_rect1: np.ndarray, grasp_rect2: np.ndarray, image_size: int, verbose: bool = False) -> float:
    """
    Expects grasp_rect1 and grasp_rect2 in the Grasp Rectangle format.
    i.e. expects both to be number arrays of shape (4, 2), with the outer dim being the
    4 corners of the rectangle and the inner dim being the x-y coordinate of a corner.
    """
    cc1, rr1 = polygon(grasp_rect1[:, 0], grasp_rect1[:, 1], (image_size, image_size))
    cc2, rr2 = polygon(grasp_rect2[:, 0], grasp_rect2[:, 1], (image_size, image_size))
    canvas = np.zeros((image_size, image_size))
    canvas[rr1, cc1] += 1
    canvas[rr2, cc2] += 1
    union = np.sum(canvas > 0)
    if union == 0:
        return 0
    intersection = np.sum(canvas == 2)
    iou_score = intersection / union

    if verbose:
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        ax = ax.flatten()
        fig.suptitle(f"IOU Score: {iou_score}")

        canvas1 = np.zeros_like(canvas)
        canvas2 = np.zeros_like(canvas)
        canvas1[rr1, cc1] += 1
        canvas2[rr2, cc2] += 1

        ax[0].set_title("Predicted")
        ax[0].imshow(canvas1)

        ax[1].set_title("Target")
        ax[1].imshow(canvas2)

        ax[2].set_title("Combined")
        ax[2].imshow(canvas)
        plt.show()

    return iou_score

def max_iou(predicted_rect: np.ndarray, target_rects: np.ndarray, image_size: int, angle_threshold: float = np.pi / 6, verbose: bool = False):
    """
    Returns maximum iou between a single predicted rect and multiple target rects.
    Only counts the iou score for rectangles with angles within angle_threshold.
    Predicted rect shape -> (4, 2)
    Target rects shape -> (N, 4, 2)
    All rects in Grasp Rectangle format.
    """
    max_iou_score = 0
    predicted_rect_angle = angle_from_grasp_rectangle(predicted_rect)
    for target_rect in target_rects:
        target_rect_angle = angle_from_grasp_rectangle(target_rect)
        if abs(predicted_rect_angle - target_rect_angle) < angle_threshold:
            curr_iou = iou(predicted_rect, target_rect, image_size, verbose=verbose)
            max_iou_score = max(max_iou_score, curr_iou)
        else:
            if verbose:
                print(f"Target angle: {target_rect_angle}, Predicted angle: {predicted_rect_angle}")
    return max_iou_score

def max_iou_bool(predicted_rect: np.ndarray, target_rects: np.ndarray, image_size: int, angle_threshold: float = np.pi / 6, verbose: bool = False, iou_threshold: float = 0.25):
    """
    Returns boolean on weather a single predicted rect has a maximum iou over threshold
    between multiple target rects
    """
    
    return max_iou(predicted_rect=predicted_rect, target_rects=target_rects, image_size=image_size, angle_threshold=angle_threshold, verbose=verbose) > iou_threshold
