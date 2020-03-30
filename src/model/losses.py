import torch
import torch.nn as nn
from torch.nn import functional as F


class MyBCELoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, output, target):
        """
        Args:
            output (torch.Tensor) (N, C): The model output.
            target (torch.LongTensor) (N, C): The data target.

        Returns:
            loss (torch.Tensor) (0): The cross entropy loss.
        """
        # target = torch.zeros_like(output).scatter_(1, target, 1.) # (N, C)
        loss = F.binary_cross_entropy_with_logits(output, target)
        return loss

    def get_name(self):
        return 'BCELoss'

class MyCrossEntropyLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, output, target):
        """
        Args:
            output (torch.Tensor) (N, C): The model output.
            target (torch.LongTensor) (N, C): The data target.

        Returns:
            loss (torch.Tensor) (0): The cross entropy loss.
        """
        target = target.squeeze()
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(output, target)
        return loss

    def get_name(self):
        return 'CELoss'