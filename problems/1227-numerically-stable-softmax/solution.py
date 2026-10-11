import torch

def softmax(t, dim):
    """Numerically stable softmax along dim.

    Args:
        t (torch.Tensor): input tensor
        dim (int): dimension along which to apply softmax

    Returns:
        torch.Tensor: tensor of same shape as t; slices along dim sum to 1
    """
    # TODO: subtract max along dim, exp, then normalize
    m = torch.max(t, dim=dim, keepdim=True).values
    shifted = t - m
    exp_t = torch.exp(shifted)
    dominator = torch.sum(exp_t, dim=dim, keepdim=True)
    return exp_t / dominator