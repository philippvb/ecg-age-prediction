import torch
import math

# maybe go to mean at some point
def mse(ages:torch.Tensor, pred_ages:torch.Tensor, weights=None, reduction=torch.sum)->torch.Tensor:
    diff = ages.flatten() - pred_ages.flatten()
    if torch.is_tensor(weights):
        loss = reduction(weights.flatten() * diff * diff)
    else:
        loss = reduction(diff * diff)
    return loss


def mae(ages:torch.Tensor, pred_ages:torch.Tensor, weights=None, reduction=torch.sum)->torch.Tensor:
    diff = ages.flatten() - pred_ages.flatten()
    if torch.is_tensor(weights):
        wmae = reduction(weights.flatten() * torch.abs(diff))
    else:
        wmae = reduction(torch.abs(diff))
    return wmae

def gaussian_nll(target: torch.Tensor, pred: torch.Tensor, pred_log_var: torch.Tensor, weights=None, reduction=torch.mean) -> torch.Tensor:
    """Computes the sum of the batch negative log-likelihoods under a normal distribution: N(target, pred, pred_var). Scaling constants are dropped.

    Args:
        target (torch.Tensor): The target
        pred (torch.Tensor): The prediction
        pred_var (torch.Tensor): The variance of the prediction
        weights: if not None, will weight the datapoints accordingly

    Returns:
        (torch.Tensor): The sum of the negative log_likelihoods over the batch
    """
    mse = torch.pow(target - pred, 2)
    exponent = torch.exp(-pred_log_var)*mse
    loss = exponent + pred_log_var
    if torch.is_tensor(weights):
        loss = weights * loss
    loss = reduction(loss)
    return loss, reduction(exponent), reduction(pred_log_var)
