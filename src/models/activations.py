import torch
import torch.nn.functional as F


def repu(x: torch.Tensor, p: float) -> torch.Tensor:
    """
    Rectified Power Unit (RePU) activation function.
    :param x :type (torch.Tensor): input tensor.
    :param p :type (float): power parameter (p > 1).
    :return torch.Tensor: Output tensor after applying RePU.
    """
    assert p >= 1
    if p <= 1:
        raise ValueError("The power parameter p must be greater than 1.")
    return F.relu(x) ** p


class RePU(torch.nn.Module):
    def __init__(self):
        super(RePU).__init__()

    def __call__(self, *args, **kwargs):
        repu(x, p)
