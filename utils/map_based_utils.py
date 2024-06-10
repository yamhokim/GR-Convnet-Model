import torch
import numpy as np
from skimage.filters import gaussian
from skimage.feature import peak_local_max

def post_process_map_output(
        q_img: torch.Tensor, 
        cos_img: torch.Tensor, 
        sin_img: torch.Tensor, 
        width_img: torch.Tensor,
        length_img: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q_img = q_img.cpu().numpy().squeeze()
    ang_img = (torch.atan2(sin_img, cos_img) / 2.0).cpu().numpy().squeeze()
    width_img = width_img.cpu().numpy().squeeze()
    width_img *= width_img.shape[-1]
    length_img = length_img.cpu().numpy().squeeze()
    length_img *= length_img.shape[-1]

    q_img = gaussian(q_img, 2.0, preserve_range=True)
    ang_img = gaussian(ang_img, 2.0, preserve_range=True)
    width_img = gaussian(width_img, 1.0, preserve_range=True)
    length_img = gaussian(length_img, 1.0, preserve_range=True)

    return q_img, ang_img, width_img, length_img


def cls_from_map(cls_map: torch.Tensor, verbose: bool = False):
    """
    Expects input tensor cls_map to be of shape [batch_size, num_classes + 1, image_size, image_size]
    cls_map[:, 0] is the confidence map
    """
    cls_map = cls_map.permute(1, 0, 2, 3)
    conf, preds = cls_map[0].round(), cls_map[1:]
    conf = conf.repeat(preds.shape[0], 1, 1, 1)
    masked_preds = conf * preds
    masked_preds = masked_preds.view(masked_preds.shape[0], masked_preds.shape[1], -1)
    masked_preds = masked_preds.mean(-1)
    if verbose:
        print(masked_preds)
    masked_preds = masked_preds.argmax(0)
    return masked_preds



def grasps_from_map(
    conf_map: np.ndarray,
    angle_map: np.ndarray,
    width_map: np.ndarray,
    length_map: np.ndarray,
    num_peaks: int,
    verbose: bool = False
    ):
    """
    Computes and returns the top num_peaks grasps from the maps.    
    A Grasp describes a bounding box in the following format
    [center_x, center_y, angle, length, width]

    Output is of shape (num_peaks, 5)
    """
    local_max = peak_local_max(conf_map, min_distance=10, threshold_abs=0.2, num_peaks=num_peaks)

    predicted_grasps = []
    for grasp_coord in local_max:
        grasp_coord = tuple(grasp_coord)
        grasp_angle = angle_map[grasp_coord]
        grasp_width = width_map[grasp_coord]
        grasp_length = length_map[grasp_coord]
        predicted_grasps.append((grasp_coord[0], grasp_coord[1], grasp_angle, grasp_width, grasp_length))

    predicted_grasps = np.array(predicted_grasps)

    if verbose:
        print(predicted_grasps)

    return predicted_grasps