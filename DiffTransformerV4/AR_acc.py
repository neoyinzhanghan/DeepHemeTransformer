import torch
import torch.nn as nn
from BMAassumptions import BMA_final_classes


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
    err_min = torch.min(err - rel_error_allowance, err - abs_error_allowance)

    # loss should be 1 if the err_max is greater than 0 and 0 otherwise
    indicator = torch.where(
        err_min < 0, torch.ones_like(err_min), torch.zeros_like(err_min)
    )

    # Reduce across dimensions: sum over logits and average over the batch
    return indicator.mean(dim=1).mean()


def custom_a_acc(g, logits, D=0.02):
    """
    Args:
        D (float): Absolute allowable error

    Returns:
        torch.Tensor: Scalar tensor representing the average L2 loss.
    """

    assert D > 0 and D < 1, "Absolute allowable error must be between 0 and 1"
    # relative error allowance
    abs_error_allowance = D * torch.ones_like(g)

    # Compute the element-wise squared error: abs(g - logits)
    err = torch.abs(g - logits)

    # one of the error allowance must be satisfied
    err_min = err - abs_error_allowance

    # loss should be 1 if the err_max is greater than 0 and 0 otherwise
    indicator = torch.where(
        err_min < 0, torch.ones_like(err_min), torch.zeros_like(err_min)
    )

    # Reduce across dimensions: sum over logits and average over the batch
    return indicator.mean(dim=1).mean()


def custom_r_acc(g, logits, d=0.2):
    """
    Args:
        d (float): Relative allowable error proportion

    Returns:
        torch.Tensor: Scalar tensor representing the average L2 loss.
    """
    assert (
        d > 0 and d < 1
    ), "Relative allowable error proportion must be between 0 and 1"
    # relative error allowance
    rel_error_allowance = d * torch.abs(g)

    # Compute the element-wise squared error: abs(g - logits)
    err = torch.abs(g - logits)

    # one of the error allowance must be satisfied
    err_min = err - rel_error_allowance

    # loss should be 1 if the err_max is greater than 0 and 0 otherwise
    indicator = torch.where(
        err_min < 0, torch.ones_like(err_min), torch.zeros_like(err_min)
    )

    # Reduce across dimensions: sum over logits and average over the batch
    return indicator.mean(dim=1).mean()


class AR_acc(nn.Module):
    """
    Custom AR (absolute and relative) accuracy function.
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


class A_acc(nn.Module):
    """
    Custom A (absolute) accuracy function.
    """

    def __init__(self):
        super(A_acc, self).__init__()

    def forward(self, g, logits, D=0.02):
        """
        Args:
            D (float): Absolute allowable error

        Returns:
            torch.Tensor: Scalar tensor representing the average L2 loss.
        """

        return custom_a_acc(g, logits, D)


class R_acc(nn.Module):
    """
    Custom R (relative) accuracy function.
    """

    def __init__(self):
        super(R_acc, self).__init__()

    def forward(self, g, logits, d=0.2):
        """
        Args:
            d (float): Relative allowable error proportion

        Returns:
            torch.Tensor: Scalar tensor representing the average L2 loss.
        """

        return custom_r_acc(g, logits, d)


def myelocytes_ar_acc(g, logits):

    # g has shape [batch_size, 9]
    # logits has shape [batch_size, 9]

    # take the first column of g and logits
    g = g[:, 0]
    logits = logits[:, 0]

    return custom_ar_acc(g, logits)


def metamyelocytes_ar_acc(g, logits):

    # g has shape [batch_size, 9]
    # logits has shape [batch_size, 9]

    # take the second column of g and logits
    g = g[:, 1]
    logits = logits[:, 1]

    return custom_ar_acc(g, logits)


def neutrophils_bands_ar_acc(g, logits):

    # g has shape [batch_size, 9]
    # logits has shape [batch_size, 9]

    # take the third column of g and logits
    g = g[:, 2]
    logits = logits[:, 2]

    return custom_ar_acc(g, logits)


def monocytes_ar_acc(g, logits):

    # g has shape [batch_size, 9]
    # logits has shape [batch_size, 9]

    # take the fourth column of g and logits
    g = g[:, 3]
    logits = logits[:, 3]

    return custom_ar_acc(g, logits)


def eosinophils_ar_acc(g, logits):

    # g has shape [batch_size, 9]
    # logits has shape [batch_size, 9]

    # take the fifth column of g and logits
    g = g[:, 4]
    logits = logits[:, 4]

    return custom_ar_acc(g, logits)


def erythroid_precursors_ar_acc(g, logits):

    # g has shape [batch_size, 9]
    # logits has shape [batch_size, 9]

    # take the sixth column of g and logits
    g = g[:, 5]
    logits = logits[:, 5]

    return custom_ar_acc(g, logits)


def lymphocytes_ar_acc(g, logits):

    # g has shape [batch_size, 9]
    # logits has shape [batch_size, 9]

    # take the seventh column of g and logits
    g = g[:, 6]
    logits = logits[:, 6]

    return custom_ar_acc(g, logits)


def plasma_cells_ar_acc(g, logits):

    # g has shape [batch_size, 9]
    # logits has shape [batch_size, 9]

    # take the eighth column of g and logits
    g = g[:, 7]
    logits = logits[:, 7]

    return custom_ar_acc(g, logits)


def blasts_and_blast_equivalents_ar_acc(g, logits):

    # g has shape [batch_size, 9]
    # logits has shape [batch_size, 9]

    # take the ninth column of g and logits
    g = g[:, 8]
    logits = logits[:, 8]

    return custom_ar_acc(g, logits)


class Class_AR_acc(nn.Module):
    """
    Custom AR (absolute and relative) accuracy function for specific BMA final classes.
    """

    def __init__(self, class_name):
        super(AR_acc, self).__init__()
        self.class_name = class_name

        assert (
            self.class_name in BMA_final_classes
        ), f"Invalid class name: {self.class_name} in Class_AR_acc. Supported classes are: {BMA_final_classes}"

    def forward(self, g, logits, d=0.2, D=0.02):
        """
        Args:
            D (float): Absolute allowable error
            d (float): Relative allowable error proportion

        Returns:
            torch.Tensor: Scalar tensor representing the average L2 loss.
        """

        if self.class_name == "myelocytes":
            return myelocytes_ar_acc(g, logits)
        elif self.class_name == "metamyelocytes":
            return metamyelocytes_ar_acc(g, logits)
        elif self.class_name == "neutrophils/bands":
            return neutrophils_bands_ar_acc(g, logits)
        elif self.class_name == "monocytes":
            return monocytes_ar_acc(g, logits)
        elif self.class_name == "eosinophils":
            return eosinophils_ar_acc(g, logits)
        elif self.class_name == "erythroid precursors":
            return erythroid_precursors_ar_acc(g, logits)
        elif self.class_name == "lymphocytes":
            return lymphocytes_ar_acc(g, logits)
        elif self.class_name == "plasma cells":
            return plasma_cells_ar_acc(g, logits)
        elif self.class_name == "blast and blast-equivalents":
            return blasts_and_blast_equivalents_ar_acc(g, logits)
        else:
            raise ValueError(
                f"Invalid class name: {self.class_name}, in Class_AR_acc. Supported classes are: {BMA_final_classes}"
            )


def myelocytes_a_acc(g, logits):

    # g has shape [batch_size, 9]
    # logits has shape [batch_size, 9]

    # take the first column of g and logits
    g = g[:, 0]
    logits = logits[:, 0]

    return custom_a_acc(g, logits)


def metamyelocytes_a_acc(g, logits):

    # g has shape [batch_size, 9]
    # logits has shape [batch_size, 9]

    # take the second column of g and logits
    g = g[:, 1]
    logits = logits[:, 1]

    return custom_a_acc(g, logits)


def neutrophils_bands_a_acc(g, logits):

    # g has shape [batch_size, 9]
    # logits has shape [batch_size, 9]

    # take the third column of g and logits
    g = g[:, 2]
    logits = logits[:, 2]

    return custom_a_acc(g, logits)


def monocytes_a_acc(g, logits):

    # g has shape [batch_size, 9]
    # logits has shape [batch_size, 9]

    # take the fourth column of g and logits
    g = g[:, 3]
    logits = logits[:, 3]

    return custom_a_acc(g, logits)


def eosinophils_a_acc(g, logits):

    # g has shape [batch_size, 9]
    # logits has shape [batch_size, 9]

    # take the fifth column of g and logits
    g = g[:, 4]
    logits = logits[:, 4]

    return custom_a_acc(g, logits)


def erythroid_precursors_a_acc(g, logits):

    # g has shape [batch_size, 9]
    # logits has shape [batch_size, 9]

    # take the sixth column of g and logits
    g = g[:, 5]
    logits = logits[:, 5]

    return custom_a_acc(g, logits)


def lymphocytes_a_acc(g, logits):

    # g has shape [batch_size, 9]
    # logits has shape [batch_size, 9]

    # take the seventh column of g and logits
    g = g[:, 6]
    logits = logits[:, 6]

    return custom_a_acc(g, logits)


def plasma_cells_a_acc(g, logits):

    # g has shape [batch_size, 9]
    # logits has shape [batch_size, 9]

    # take the eighth column of g and logits
    g = g[:, 7]
    logits = logits[:, 7]

    return custom_a_acc(g, logits)


def blasts_and_blast_equivalents_a_acc(g, logits):

    # g has shape [batch_size, 9]
    # logits has shape [batch_size, 9]

    # take the ninth column of g and logits
    g = g[:, 8]
    logits = logits[:, 8]

    return custom_a_acc(g, logits)


class Class_A_acc(nn.Module):
    """
    Custom A (absolute) accuracy function for specific BMA final classes.
    """

    def __init__(self, class_name):
        super(Class_A_acc, self).__init__()
        self.class_name = class_name

        assert (
            self.class_name in BMA_final_classes
        ), f"Invalid class name: {self.class_name} in Class_A_acc. Supported classes are: {BMA_final_classes}"

    def forward(self, g, logits, D=0.02):
        """
        Args:
            D (float): Absolute allowable error

        Returns:
            torch.Tensor: Scalar tensor representing the average L2 loss.
        """

        if self.class_name == "myelocytes":
            return myelocytes_a_acc(g, logits)
        elif self.class_name == "metamyelocytes":
            return metamyelocytes_a_acc(g, logits)
        elif self.class_name == "neutrophils/bands":
            return neutrophils_bands_a_acc(g, logits)
        elif self.class_name == "monocytes":
            return monocytes_a_acc(g, logits)
        elif self.class_name == "eosinophils":
            return eosinophils_a_acc(g, logits)
        elif self.class_name == "erythroid precursors":
            return erythroid_precursors_a_acc(g, logits)
        elif self.class_name == "lymphocytes":
            return lymphocytes_a_acc(g, logits)
        elif self.class_name == "plasma cells":
            return plasma_cells_a_acc(g, logits)
        elif self.class_name == "blast and blast-equivalents":
            return blasts_and_blast_equivalents_a_acc(g, logits)
        else:
            raise ValueError(
                f"Invalid class name: {self.class_name}, in Class_A_acc. Supported classes are: {BMA_final_classes}"
            )


def myelocytes_r_acc(g, logits):

    # g has shape [batch_size, 9]
    # logits has shape [batch_size, 9]

    # take the first column of g and logits
    g = g[:, 0]
    logits = logits[:, 0]

    return custom_r_acc(g, logits)


def metamyelocytes_r_acc(g, logits):

    # g has shape [batch_size, 9]
    # logits has shape [batch_size, 9]

    # take the second column of g and logits
    g = g[:, 1]
    logits = logits[:, 1]

    return custom_r_acc(g, logits)


def neutrophils_bands_r_acc(g, logits):

    # g has shape [batch_size, 9]
    # logits has shape [batch_size, 9]

    # take the third column of g and logits
    g = g[:, 2]
    logits = logits[:, 2]

    return custom_r_acc(g, logits)


def monocytes_r_acc(g, logits):

    # g has shape [batch_size, 9]
    # logits has shape [batch_size, 9]

    # take the fourth column of g and logits
    g = g[:, 3]
    logits = logits[:, 3]

    return custom_r_acc(g, logits)


def eosinophils_r_acc(g, logits):

    # g has shape [batch_size, 9]
    # logits has shape [batch_size, 9]

    # take the fifth column of g and logits
    g = g[:, 4]
    logits = logits[:, 4]

    return custom_r_acc(g, logits)


def erythroid_precursors_r_acc(g, logits):

    # g has shape [batch_size, 9]
    # logits has shape [batch_size, 9]

    # take the sixth column of g and logits
    g = g[:, 5]
    logits = logits[:, 5]

    return custom_r_acc(g, logits)


def lymphocytes_r_acc(g, logits):

    # g has shape [batch_size, 9]
    # logits has shape [batch_size, 9]

    # take the seventh column of g and logits
    g = g[:, 6]
    logits = logits[:, 6]

    return custom_r_acc(g, logits)


def plasma_cells_r_acc(g, logits):

    # g has shape [batch_size, 9]
    # logits has shape [batch_size, 9]

    # take the eighth column of g and logits
    g = g[:, 7]
    logits = logits[:, 7]

    return custom_r_acc(g, logits)


def blasts_and_blast_equivalents_r_acc(g, logits):

    # g has shape [batch_size, 9]
    # logits has shape [batch_size, 9]

    # take the ninth column of g and logits
    g = g[:, 8]
    logits = logits[:, 8]

    return custom_r_acc(g, logits)


class Class_R_acc(nn.Module):
    """
    Custom R (relative) accuracy function for specific BMA final classes.
    """

    def __init__(self, class_name):
        super(Class_R_acc, self).__init__()
        self.class_name = class_name

        assert (
            self.class_name in BMA_final_classes
        ), f"Invalid class name: {self.class_name} in Class_R_acc. Supported classes are: {BMA_final_classes}"

    def forward(self, g, logits, d=0.2):
        """
        Args:
            d (float): Relative allowable error proportion

        Returns:
            torch.Tensor: Scalar tensor representing the average L2 loss.
        """

        if self.class_name == "myelocytes":
            return myelocytes_r_acc(g, logits)
        elif self.class_name == "metamyelocytes":
            return metamyelocytes_r_acc(g, logits)
        elif self.class_name == "neutrophils/bands":
            return neutrophils_bands_r_acc(g, logits)
        elif self.class_name == "monocytes":
            return monocytes_r_acc(g, logits)
        elif self.class_name == "eosinophils":
            return eosinophils_r_acc(g, logits)
        elif self.class_name == "erythroid precursors":
            return erythroid_precursors_r_acc(g, logits)
        elif self.class_name == "lymphocytes":
            return lymphocytes_r_acc(g, logits)
        elif self.class_name == "plasma cells":
            return plasma_cells_r_acc(g, logits)
        elif self.class_name == "blast and blast-equivalents":
            return blasts_and_blast_equivalents_r_acc(g, logits)
        else:
            raise ValueError(
                f"Invalid class name: {self.class_name}, in Class_R_acc. Supported classes are: {BMA_final_classes}"
            )
