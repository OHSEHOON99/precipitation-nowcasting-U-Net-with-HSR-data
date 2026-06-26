import torch
import torch.nn as nn

from utils import *


# Define Loss function
class BalancedMSEMAE(nn.Module):
    def __init__(self, mse_weight=1.0, mae_weight=1.0, normal_loss_scale_factor=0.00005):
        """
        Initializes the BalancedMSEMAE module.

        Parameters:
        - mse_weight (float): Weight for the Mean Squared Error (MSE) term in the loss function.
        - mae_weight (float): Weight for the Mean Absolute Error (MAE) term in the loss function.
        - normal_loss_scale_factor (float): Scaling factor applied to the entire loss to control its magnitude.
        """
        super().__init__()
        self.normal_loss_scale_factor = normal_loss_scale_factor
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight

    def forward(self, labels_pred, labels):
        """
        Calculates the balanced MSE-MAE loss.

        Parameters:
        - labels_pred (torch.Tensor): Predicted labels.
        - labels (torch.Tensor): Ground truth labels.

        Returns:
        - loss (torch.Tensor): Computed loss.
        """
        balancing_weights = (1, 1, 2, 5, 10, 30)  # Balancing weights for different ranges
        thresholds = [dBZ_to_pixel(rfrate_to_dBZ(ele)) for ele in [0.5, 2, 5, 10, 30]]

        # Initialize weights tensor with balancing_weights[0], and ensure it matches the device of labels_pred
        weights = torch.ones_like(labels_pred, device=labels_pred.device) * balancing_weights[0]

        # Efficient threshold comparison using vectorized operations
        for i, threshold in enumerate(thresholds):
            weights += (balancing_weights[i + 1] - balancing_weights[i]) * (labels >= threshold).float()

        # Compute weighted MSE and MAE
        mse = torch.sum(weights * ((labels_pred - labels) ** 2), dim=(1, 2, 3))  # Sum over (C, H, W)
        mae = torch.sum(weights * torch.abs(labels_pred - labels), dim=(1, 2, 3))  # Sum over (C, H, W)

        # Compute final loss with scaling factor and weighted terms
        loss = self.normal_loss_scale_factor * (self.mse_weight * torch.mean(mse) + self.mae_weight * torch.mean(mae))

        return loss