import numpy as np
import sys
sys.path.append(".")

def angle_from_grasp_rectangle(grasp_rect: np.ndarray) -> float:
    """
    Computes the orientation angle of a grasp rectangle
    """
    dx = grasp_rect[1, 0] - grasp_rect[0, 0]
    dy = grasp_rect[1, 1] - grasp_rect[0, 1]
    return (np.arctan2(-dy, dx) + np.pi / 2) % np.pi - np.pi / 2


def grasp_rect_from_grasps(grasps: np.ndarray) -> np.ndarray:
    """
    Expects inputs to be in the Grasp format with shape (N, 5).
    Each of the N in the first dim represent a different grasp, with each being in the order
    [center_x, center_y, angle, length, width]
    """
    y, x, angle, length, width = grasps.transpose()
    xo = np.cos(angle)
    yo = np.sin(angle)

    ya = y + length / 2 * yo
    xa = x - length / 2 * xo
    yb = y - length / 2 * yo
    xb = x + length / 2 * xo

    y1, x1 = ya - width / 2 * xo, xa - width / 2 * yo
    y2, x2 = yb - width / 2 * xo, xb - width / 2 * yo
    y3, x3 = yb + width / 2 * xo, xb + width / 2 * yo
    y4, x4 = ya + width / 2 * xo, xa + width / 2 * yo

    p1 = np.stack([x1, y1], 1)
    p2 = np.stack([x2, y2], 1)
    p3 = np.stack([x3, y3], 1)
    p4 = np.stack([x4, y4], 1)

    output_arr = np.stack([p1, p2, p3, p4], 1)
    return output_arr


def check_grasp_success(
    predicted_rects: np.ndarray,
    target_rects: np.ndarray,
    image_size: int,
    angle_threshold: float,
    iou_threshold: float,
    verbose: bool = False
    ):
    """
    Given an array list of predicted grasp rectangles and target grasp rectangles, outputs
    if max_iou between at least one predicted grasp rectangle > threshold.
    """
    from metrics.iou import max_iou # placed here due to circular import problems

    for rect in predicted_rects:
        iou_score = max_iou(
            predicted_rect=rect, 
            target_rects=target_rects,
            angle_threshold=angle_threshold,
            image_size=image_size,
            verbose=verbose
        )
        if iou_score > iou_threshold:
            return True
    return False