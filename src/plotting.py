from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import torch

@torch.no_grad()
def plot_calibration(model, test_loader, axs, device):
    errors = []
    confidences = []
    for index, (data, target) in enumerate(test_loader):
        target = target.to(device)
        data=data.to(device)        
        pred_ages, pred_ages_log_var = model(data)
        error = torch.square(pred_ages - target)
        errors.append(error)
        confidences.append(torch.exp(pred_ages_log_var))
    
    errors = torch.unsqueeze(torch.cat(errors), dim=-1).cpu().numpy()
    confidences = torch.unsqueeze(torch.cat(confidences), dim=-1).cpu().numpy()
    axs.scatter(confidences, errors)
