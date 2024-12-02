import torch
import torch.nn as nn


class MyL2Loss(nn.Module):
    """
    Custom L2 loss function for PyTorch Lightning: computes the mean squared error
    between target vectors and predictions.
    """

    def __init__(self):
        super(MyL2Loss, self).__init__()

    def forward(self, g, logits):
        """
        Args:
            g (torch.Tensor): Target vector of shape [b, d], where b is the batch size
                              and d is the dimension of the logit.
            logits (torch.Tensor): Logit vector of shape [b, d], where b is the batch size
                                   and d is the dimension of the logit.

        Returns:
            torch.Tensor: Scalar tensor representing the average L2 loss.
        """
        # Compute the element-wise squared error: (g - logits)^2
        loss = (g - logits) ** 2

        # Reduce across dimensions: sum over logits and average over the batch
        return loss.sum(dim=1).mean()


def custom_l2_loss(g, logits):
    """
    Custom L2 loss function: computes the mean squared error between target vectors
    and predictions, and averages across the batch.

    Args:
        g (torch.Tensor): Target vector of shape [b, d], where b is the batch size
                          and d is the dimension of the logit.
        logits (torch.Tensor): Prediction vector of shape [b, d], where b is the batch size
                               and d is the dimension of the logit.

    Returns:
        torch.Tensor: Scalar tensor representing the average L2 loss.
    """
    # Compute the element-wise squared error: (g - logits)^2
    loss = (g - logits) ** 2

    # Reduce across dimensions: sum over logits and average over the batch
    loss = loss.sum(dim=1).mean()

    return loss
