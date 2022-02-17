import torch
import math

# maybe go to mean at some point
def mse(ages:torch.Tensor, pred_ages:torch.Tensor, weights=None)->torch.Tensor:
    diff = ages.flatten() - pred_ages.flatten()
    if torch.is_tensor(weights):
        loss = torch.sum(weights.flatten() * diff * diff)
    else:
        loss = torch.sum(diff * diff)
    return loss


def mae(ages:torch.Tensor, pred_ages:torch.Tensor, weights=None)->torch.Tensor:
    diff = ages.flatten() - pred_ages.flatten()
    if torch.is_tensor(weights):
        wmae = torch.sum(weights.flatten() * torch.abs(diff))
    else:
        wmae = torch.sum(torch.abs(diff))
    return wmae

def gaussian_nll(target: torch.Tensor, pred: torch.Tensor, pred_log_var: torch.Tensor, weights=None, cutoff=torch.Tensor([0.01, 100])) -> torch.Tensor:
    """Computes the sum of the batch negative log-likelihoods under a normal distribution: N(target, pred, pred_var). Scaling constants are dropped.

    Args:
        target (torch.Tensor): The target
        pred (torch.Tensor): The prediction
        pred_var (torch.Tensor): The variance of the prediction
        weights: if not None, will weight the datapoints accordingly

    Returns:
        (torch.Tensor): The sum of the negative log_likelihoods over the batch
    """
#    pred_log_var = torch.minimum(torch.maximum(torch.log(cutoff[0]), pred_log_var), torch.log(cutoff[1]))
    # we can drop the 0.5 since it is both in the exponent and in the sqrt for the first term
    loss = target.shape[0] * pred_log_var + torch.square(pred - target) / torch.exp(pred_log_var)
    if torch.is_tensor(weights):
        return (weights * loss).mean()
    else:
        return loss.mean()
