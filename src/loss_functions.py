import torch

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

def gaussian_nll(target: torch.Tensor, pred: torch.Tensor, pred_var: torch.Tensor, weights=None) -> torch.Tensor:
    """Computes the sum of the batch negative log-likelihoods under a normal distribution: N(target, pred, pred_var). Scaling constants are dropped.

    Args:
        target (torch.Tensor): The target
        pred (torch.Tensor): The prediction
        pred_var (torch.Tensor): The variance of the prediction
        weights: if not None, will weight the datapoints accordingly

    Returns:
        (torch.Tensor): The sum of the negative log_likelihoods over the batch
    """
    loss = torch.square(pred - target) / pred_var + torch.log(pred_var)
    if torch.is_tensor(weights):
        return (weights * loss).sum()
    else:
        return loss.sum()