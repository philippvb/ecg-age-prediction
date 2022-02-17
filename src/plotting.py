from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import torch

@torch.no_grad()
def plot_calibration(model, test_loader, axs, device):
    errors = []
    confidences = []
    for index, (data, target) in enumerate(test_loader):
        target = target.to(device)
        data = data.to(device)        
        pred_ages, pred_ages_var = model(data)
        error = torch.abs(pred_ages - target)
        errors.append(error)
        confidences.append(pred_ages_var.sqrt())
    
    errors = torch.squeeze(torch.cat(errors)).cpu().numpy()
    confidences = torch.squeeze(torch.cat(confidences)).cpu().numpy()
    axs.scatter(confidences, errors)
