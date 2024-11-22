import torch.nn as nn
import torch.nn as nn
import numpy
import torch


class RegLoss(nn.Module):
    """
    Computes a regularization loss (L1 or L2) over all model parameters.
    """

    def __init__(self, loss_weight=1.0):
        """
        Args:
            power (int): 1 for L1 loss, 2 for L2 loss.
            loss_weight (float): Weight to scale the loss.
        """
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, model):
        total_loss = 0.0
        total_params = 0

        for param in model.parameters():
            if param.requires_grad:  # Include only trainable parameters
                total_loss += param.square().sum()  # Sum of squared values
                total_params += param.numel()
        return (1.0 / (total_loss / total_params)) * self.loss_weight