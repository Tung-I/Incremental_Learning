import torch
import torch.nn as nn
from torch.nn import functional as F


class DistilledCrossEntropyLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, output, target, meta_data=None):
        """
        Args:
            output (torch.Tensor) (N, C): The model output.
            target (torch.LongTensor) (N, C): The data target.

        Returns:
            loss (torch.Tensor) (0): The cross entropy loss.
        """
        loss = F.binary_cross_entropy_with_logits(output, target)
        return loss
