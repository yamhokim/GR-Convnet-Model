import torch
from torch import nn
from .double_log import DoubleLogLoss

class MapLoss:

    def __init__(self):
        self.double_log = DoubleLogLoss(mean_reduction=False)
        self.bce = nn.BCELoss()

    def __call__(self, predicted_map: torch.Tensor, target_map: torch.Tensor) -> torch.Tensor:
        """
        predicted_map and target_map are of shape [batch_size, n_channels, img_size, img_size].
        indexing with [:, 0, :, :] represents confidence maps for both predicted and target.
        """
        # Reshape to [n_channels, batch_size, img_size, img_size]
        predicted_map = predicted_map.permute(1, 0, 2, 3)
        target_map = target_map.permute(1, 0, 2, 3)

        # Computing a confidence map loss with the first channel
        confidence_loss = self.bce(predicted_map[0], target_map[0])

        # Computing the grasp/cls losses with remaining channels
        grasp_cls_loss = self.double_log(predicted_map[1:], target_map[1:])

        # Valid pixels for grasp_cls_loss are those where the target confidence map is not 0, 
        # since these are pixels which belong to the object and not the background.
        valid_pixels = target_map[0] != 0
        valid_pixels = valid_pixels.unsqueeze(0).repeat(predicted_map.shape[0] - 1, 1, 1, 1)
        grasp_cls_loss = (grasp_cls_loss * valid_pixels).mean()
        return confidence_loss + grasp_cls_loss * 2