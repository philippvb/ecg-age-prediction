from tqdm import tqdm
from src.dataloader import BatchDataloader
import torch
from src.loss_functions import *

def eval(model, ep, dataload:BatchDataloader, device, probabilistic=False):
    model.eval()
    n_entries = 0

    eval_desc = "Epoch {:2d}: valid - Loss(WMSE): {:.6f}"
    eval_bar = tqdm(initial=0, leave=True, total=len(dataload),
                    desc=eval_desc.format(ep, 0, 0), position=0)

    # tracking
    total_mse = 0
    total_wmse = 0
    total_wmae = 0

    for traces, ages, weights in dataload:
        traces = traces.transpose(1,2)
        traces, ages, weights = traces.to(device), ages.to(device), weights.to(device)
        with torch.no_grad():
            # Forward pass
            if probabilistic:
                pred_ages, pred_ages_var = model(traces)
            else:
                pred_ages = model(traces)
            total_wmse += mse(ages, pred_ages, weights=weights, reduction=torch.sum).cpu().numpy()
            total_mse += mse(ages, pred_ages, weights=None, reduction=torch.sum).cpu().numpy()
            total_wmae += mae(ages, pred_ages, weights, reduction=torch.sum).cpu().numpy()
            # Update outputs
            bs = len(traces)
            n_entries += bs
            # Print result
            eval_bar.desc = eval_desc.format(ep, total_wmse / n_entries)
            eval_bar.update(1)

    eval_bar.close()
    return total_mse / n_entries, total_wmse / n_entries, total_wmae /n_entries