from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import torch

@torch.no_grad()
def plot_calibration(model, test_loader, axs, device, data_noise=0, log_scale=False):
    errors = []
    confidences = []
    for index, (data, target) in enumerate(test_loader):
        target = target.to(device)
        data = data.to(device)        
        pred_ages, pred_ages_var = model(data)
        if log_scale:
            pred_ages_var = pred_ages_var.exp()
        error = torch.abs(pred_ages - target)
        errors.append(error)
        confidences.append(pred_ages_var)
    
    errors = torch.squeeze(torch.cat(errors)).cpu().numpy()
    confidences = torch.squeeze(torch.cat(confidences)).cpu()
    confidences = torch.sqrt(confidences + data_noise).numpy() # the total std deviation is given by sqrt(pred_var + data_var)
    axs.scatter(confidences, errors)
