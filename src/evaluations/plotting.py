from typing import Tuple
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from src.dataset.dataloader import BatchDataloader
from torch import nn
import torch

SCATTER_CONFIG = {"color":"black", "s":1}

def forward_summary(dataset:BatchDataloader, model:torch.nn.Module, error_fun=torch.nn.MSELoss, prob=True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    errors_list = torch.empty(1)
    ages_list = torch.empty(1)
    pred_log_var_list = torch.empty(1)
    pred_list = torch.empty(1)
    for data, target, _ in dataset:
        pred = model(data)
        if prob:
            pred_log_var = pred[1]
            pred = pred[0] # drop var if probabilistic model
            pred_log_var_list = torch.cat((pred_log_var_list, pred_log_var.squeeze().cpu()), dim=0)
        errors = error_fun(pred, target)
        errors_list = torch.cat((errors_list, errors.squeeze().cpu()), dim=0)
        ages_list = torch.cat((ages_list, target.squeeze().cpu()), dim=0)
        pred_list = torch.cat((pred_list, pred.squeeze().cpu()), dim=0)
    return pred_list, pred_log_var_list, errors_list, ages_list

def remove_outliers(x:torch.Tensor, *tensors, n:int) -> torch.Tensor:
    kth_smallest, _ = torch.kthvalue(x, n)
    kth_largest, _ = torch.kthvalue(x, len(x)-n)
    mask =  torch.logical_and(x < kth_largest, x > kth_smallest)
    return [torch.masked_select(x, mask)] + [torch.masked_select(t, mask) for t in tensors] # apply mask to all


@torch.no_grad()
def plot_calibration(dataset:BatchDataloader, model:nn.Module, axs:Axes, error_fun=torch.nn.MSELoss, plot_parameters=SCATTER_CONFIG):
    pred, pred_log_var, errors, ages = forward_summary(dataset, model, error_fun, prob=True)
    errors, pred_log_var = remove_outliers(errors, pred_log_var, n=100)
    axs.scatter(pred_log_var.exp().cpu(), errors.cpu(), **plot_parameters)
    axs.set_xlabel("Predicted variance")
    axs.set_ylabel("Error")

@torch.no_grad()
def plot_calibration_laplace(dataset:BatchDataloader, model:nn.Module, axs:Axes, error_fun=torch.nn.MSELoss, plot_parameters=SCATTER_CONFIG):
    pred, pred_var, errors, ages = forward_summary(dataset, model, error_fun, prob=True)
    errors, pred_var = remove_outliers(errors, pred_var, n=100)
    var = pred_var.cpu() + model.sigma_noise.item()**2 # add the datanoise if the model
    axs.scatter(var, errors.cpu(),  **plot_parameters)
    axs.set_xlabel("Predicted variance")
    axs.set_ylabel("Error")

@torch.no_grad()
def plot_age_vs_error(dataset:BatchDataloader, model:nn.Module, axs:Axes, error_fun=torch.nn.MSELoss, prob=True, plot_parameters=SCATTER_CONFIG):
    pred, pred_log_var, errors, ages = forward_summary(dataset, model, error_fun, prob=prob)
    errors, ages = remove_outliers(errors, ages, n=100)
    axs.scatter(ages.cpu(), errors.cpu(), **plot_parameters)
    axs.set_xlabel("target Age")
    axs.set_ylabel("Error")

@torch.no_grad()
def plot_predicted_age_vs_error(dataset:BatchDataloader, model:nn.Module, axs:Axes, error_fun=torch.nn.MSELoss, prob=True, plot_parameters=SCATTER_CONFIG):
    pred, pred_log_var, errors, ages = forward_summary(dataset, model, error_fun, prob=prob)
    errors, pred = remove_outliers(errors, pred, n=100)
    axs.scatter(pred.cpu(), errors.cpu(), **plot_parameters)
    axs.set_xlim(25, 100)
    # axs.set_ylim(0, errors.max())
    axs.set_xlabel("predicted Age")
    axs.set_ylabel("Error")

def plot_summary(axs, summary_dict:dict):
    textstr = "Summary\n"
    textstr += "\n".join([f"{key}: {value}" for key, value in summary_dict.items()])
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    axs.text(0.05, 0.95, textstr, transform=axs.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)