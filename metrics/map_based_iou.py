import torch
import numpy as np

import sys
sys.path.append(".")

from utils.map_based_utils import post_process_map_output, grasps_from_map
from utils.grasp_utils import grasp_rect_from_grasps, check_grasp_success

def map_based_iou(
    conf_map: torch.Tensor,
    cos_map: torch.Tensor,
    sin_map: torch.Tensor,
    width_map: torch.Tensor,
    length_map: torch.Tensor,
    target_grasp_rects: torch.Tensor,
    angle_threshold: float = np.pi / 6,
    iou_threshold: float = 0.25,
    num_peaks: int = 3,
    verbose: bool = False
    ) -> bool:
    """ 
    Returns a boolean value prediction whether the predicted grasps as successful or not
    """
    target_grasp_rects = target_grasp_rects.numpy()
    image_size = conf_map.shape[-1]

    conf, angle, width, length = post_process_map_output(conf_map, cos_map, sin_map, width_map, length_map)
    predicted_grasps = grasps_from_map(conf, angle, width, length, num_peaks=num_peaks, verbose=verbose)
    if predicted_grasps.size == 0:
        return False
    predicted_grasp_rects = grasp_rect_from_grasps(predicted_grasps)
    return check_grasp_success(predicted_grasp_rects, target_grasp_rects, image_size, angle_threshold, iou_threshold, verbose=verbose)