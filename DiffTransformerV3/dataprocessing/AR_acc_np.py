import numpy as np

def custom_ar_acc_np(g, logits, d=0.2, D=0.02):
    """
    Custom L2 loss function using NumPy: computes the mean squared error between
    target vectors and predictions, and averages across the batch.

    Args:
        g (np.ndarray): Target vector of shape [b, d], where b is the batch size
                        and d is the dimension of the logit.
        logits (np.ndarray): Prediction vector of shape [b, d], where b is the batch size
                             and d is the dimension of the logit.
        d (float): Relative allowable error proportion. Must be between 0 and 1.
        D (float): Absolute allowable error. Must be between 0 and 1.

    Returns:
        float: Scalar value representing the average L2 loss.
    """
    assert 0 < d < 1, "Relative allowable error proportion must be between 0 and 1"
    assert 0 < D < 1, "Absolute allowable error must be between 0 and 1"
    
    # Relative error allowance
    rel_error_allowance = d * np.abs(g)

    # Absolute error allowance
    abs_error_allowance = D * np.ones_like(g)

    # Compute the element-wise absolute error: abs(g - logits)
    err = np.abs(g - logits)

    # Compute the max allowance violation
    err_max = np.maximum(err - rel_error_allowance, err - abs_error_allowance)

    # Loss is 1 if err_max > 0, else 0
    loss = np.where(err_max > 0, np.ones_like(err_max), np.zeros_like(err_max))

    # Reduce across dimensions: sum over logits and average over the batch
    loss = loss.mean(axis=1).mean()

    return loss
