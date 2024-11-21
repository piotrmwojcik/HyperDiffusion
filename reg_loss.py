import torch.nn as nn

import torch.nn as nn

def reg_loss(tensor, power=1):
    """
    Compute regularization loss for a given tensor.
    """
    return tensor.abs().mean() if power == 1 \
        else (tensor.abs() ** power).mean()


class RegLoss(nn.Module):
    """
    Computes a regularization loss (L1 or L2) over all model parameters.
    """

    def __init__(self, power=1, loss_weight=1.0):
        """
        Args:
            power (int): 1 for L1 loss, 2 for L2 loss.
            loss_weight (float): Weight to scale the loss.
        """
        super().__init__()
        self.power = power
        self.loss_weight = loss_weight

    def forward(self, model):
        """
        Compute the regularization loss for all parameters in the model.

        Args:
            model (nn.Module): The model whose parameters will be regularized.

        Returns:
            torch.Tensor: The regularization loss value.
        """
        total_loss = 0.0
        for param in model.parameters():
            if param.requires_grad:  # Only include trainable parameters
                total_loss += reg_loss(param, power=self.power)
        return total_loss * self.loss_weight