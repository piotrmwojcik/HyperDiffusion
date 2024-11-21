import torch.nn as nn

def reg_loss(tensor, power=1):
    return tensor.abs().mean() if power == 1 \
        else (tensor.abs() ** power).mean()


class RegLoss(nn.Module):

    def __init__(self,
                 power=1,
                 loss_weight=1.0):
        super().__init__()
        self.power = power
        self.loss_weight = loss_weight

    def forward(self, tensor, weight=None, avg_factor=None):
        return reg_loss(
            tensor, power=self.power,
            weight=weight, avg_factor=avg_factor) * self.loss_weight