import torch


def custom_cross_entropy_loss(g, p):
    """
    Custom cross-entropy loss function: computes E_g[-log(p)] and averages across the batch.

    Args:
        g (torch.Tensor): Target vector of shape [b, d], where b is the batch size and d is the dimension of the logit.
        p (torch.Tensor): Probability vector of shape [b, d], where b is the batch size and d is the dimension of the logit.
                          Typically, p should be the output of a softmax operation.

    Returns:
        torch.Tensor: Scalar tensor representing the average cross-entropy loss.
    """
    # Ensure probabilities are clipped to avoid numerical instability with log(0)
    p = torch.clamp(p, min=1e-12)

    # Compute the element-wise loss: -g * log(p)
    loss = -g * torch.log(p)

    # Reduce across dimensions: sum over logits and average over the batch
    loss = loss.sum(dim=1).mean()

    return loss
