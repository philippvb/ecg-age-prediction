from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import torch

@torch.no_grad()
def plot_calibration(model, test_loader, axs):
    errors = []
    confidences = []
    for index, (data, target) in enumerate(test_loader):
        pred_ages, pred_ages_log_var = model(data)
        error = torch.square(pred_ages - target)
        errors.append(error)
        confidences.append(torch.exp(pred_ages_log_var))
    
    errors = torch.unsqueeze(torch.cat(errors), dim=-1)
    confidences = torch.unsqueeze(torch.cat(confidences), dim=-1)
    axs.scatter(confidences, errors)