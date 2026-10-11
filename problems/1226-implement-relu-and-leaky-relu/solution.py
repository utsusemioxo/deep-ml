import torch

def relu(t):
    """Element-wise ReLU: max(0, t).

    Args:
        t (torch.Tensor): input tensor

    Returns:
        torch.Tensor: activated tensor
    """
    # TODO: implement with pure torch ops
    y = torch.zeros_like(t)
    return torch.maximum(t, y)

def leaky_relu(t, slope=0.01):
    """Element-wise Leaky ReLU with given negative slope.

    Args:
        t (torch.Tensor): input tensor
        slope (float): slope for negative values

    Returns:
        torch.Tensor: activated tensor
    """
    # TODO: implement with pure torch ops
    return torch.where(t > 0, t, t * slope)
