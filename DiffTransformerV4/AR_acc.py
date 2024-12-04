import torch
import torch.nn as nn


def custom_ar_acc(g, logits, d=0.2, D=0.02):
    """
    Args:
        D (float): Absolute allowable error
        d (float): Relative allowable error proportion

    Returns:
        torch.Tensor: Scalar tensor representing the average L2 loss.
    """
    assert (
        d > 0 and d < 1
    ), "Relative allowable error proportion must be between 0 and 1"
    assert D > 0 and D < 1, "Absolute allowable error must be between 0 and 1"
    # relative error allowance
    rel_error_allowance = d * torch.abs(g)

    abs_error_allowance = D * torch.ones_like(g)

    # Compute the element-wise squared error: abs(g - logits)
    err = torch.abs(g - logits)

    # one of the error allowance must be satisfied
    err_max = torch.min(err - rel_error_allowance, err - abs_error_allowance)

    # loss should be 1 if the err_max is greater than 0 and 0 otherwise
    indicator = torch.where(
        err_max < 0, torch.ones_like(err_max), torch.zeros_like(err_max)
    )

    # Reduce across dimensions: sum over logits and average over the batch
    return indicator.mean(dim=1).mean()


class AR_acc(nn.Module):
    """
    Custom L2 loss function for PyTorch Lightning: computes the mean squared error
    between target vectors and predictions.
    """

    def __init__(self):
        super(AR_acc, self).__init__()

    def forward(self, g, logits, d=0.2, D=0.02):
        """
        Args:
            D (float): Absolute allowable error
            d (float): Relative allowable error proportion

        Returns:
            torch.Tensor: Scalar tensor representing the average L2 loss.
        """

        return custom_ar_acc(g, logits, d, D)
