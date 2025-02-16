
# Imports
import sys, os
sys.path.append(os.getcwd())
from src.models.resnet import ResNet1d
from tqdm import tqdm
import torch
import os
import json
import numpy as np
import argparse
from warnings import warn
import pandas as pd
from src.dataset.dataloader import *
from laplace import Laplace
from src.loss_functions import mse, mae
import matplotlib.pyplot as plt
from src.evaluations.plotting import *



if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--mdl',
                        help='folder containing model.')
    args, unk = parser.parse_known_args()
    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    config = os.path.join(args.mdl, 'args.json')
    with open(config, 'r') as f:
        config_dict = json.load(f)
    print(config_dict)
    # Get model
    N_LEADS = config_dict["n_leads"]
    model = ResNet1d(input_dim=(N_LEADS, config_dict['seq_length']),
                     blocks_dim=list(zip(config_dict['net_filter_size'], config_dict['net_seq_lengh'])),
                     n_classes=1,
                     kernel_size=config_dict['kernel_size'],
                     dropout_rate=config_dict['dropout_rate'])
    model.load(args.mdl)
    model = model.to(device)

    
    print("Loading data")
    # Get traces
    # how much of our dataset we actually use, on a scale from 0 to 1
    if config_dict["bianca"]:
        train_loader, valid_loader = load_dset_swedish(config_dict, use_weights=False, device=device)
    else:
        train_loader, valid_loader = load_dset_brazilian(config_dict, use_weights=False, device=device, map_to_swedish=config_dict["n_leads"]==8)
    valid_dataset_size =  valid_loader.get_size()
    train_loader.format_Laplace()
    valid_loader.format_Laplace()

    summary = {}
    summary["dataset size"] = valid_dataset_size

    print("Estimating Laplace model")
    # add Laplace
    laplace_model = Laplace(model, "regression", subset_of_weights='last_layer', hessian_structure='full')
    laplace_model.fit(train_loader)

    print("Estimating hyperparameters")
    log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
    hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)
    for i in tqdm(range(10)):
        hyper_optimizer.zero_grad()
        neg_marglik = - laplace_model.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
        neg_marglik.backward()
        hyper_optimizer.step()
    print("The estimated data noise (std deviation) is:", laplace_model.sigma_noise.item())
    summary["Data noise"] = laplace_model.sigma_noise.item()


    with torch.no_grad():

        total_mse = 0
        total_wmse = 0
        total_mae = 0
        for data, ages, weights in valid_loader:
            prediction = laplace_model(data)[0]
            total_mse += mse(ages, prediction).cpu()
            total_wmse += mse(ages, prediction, weights).cpu()
            total_mae += mae(ages, prediction).cpu()
        total_mse /= valid_dataset_size
        total_wmse /= valid_dataset_size
        total_mae /= valid_dataset_size
        print(f"MSE on validation set is: {total_mse}")
        print(f"Sqrt of MSE on validation set is: {total_mse.sqrt()}")
        print(f"Weighted MSE on validation set is: {total_wmse}")
        print(f"MAE on validation set is: {total_mae}")
        summary["MSE"] = total_mse
        summary["WMSE"] = total_wmse
        summary["MAE"] = total_mae        

        var = torch.tensor([0.0])
        for data, ages, _ in valid_loader:
            data = data.to(device)
            out_mean, out_var = laplace_model(data)
            var += out_var.cpu().sum()
        var/= valid_dataset_size
        print("Mean std is:", var.sqrt().item())
        summary["Mean std"] = var.sqrt()

        for noise in [0.1, 1, 10]:
            ood_var = torch.tensor([0.0])
            for data, ages, _ in valid_loader:
                data += noise * torch.randn_like(data)
                data = data.to(device)
                out_mean, out_var = laplace_model(data)
                ood_var += out_var.cpu().sum()
            ood_var/= valid_dataset_size
            print(f"Mean std with noise of {noise} is:", ood_var.sqrt().item())
            summary[f"Mean std, noise {noise}"] = ood_var.sqrt()

        ood_var = torch.tensor([0.0])
        for data, ages, _ in valid_loader:
            data = torch.flip(data, dims=[-1])
            data = data.to(device)
            out_mean, out_var = laplace_model(data)
            ood_var += out_var.cpu().sum()
        ood_var/= valid_dataset_size
        print(f"Mean std for flipped is:", ood_var.sqrt().item())
        summary[f"Mean std, flipped {noise}"] = ood_var.sqrt()

    fig, axs = plt.subplots(2,2, figsize=(10, 10))
    axs = axs.flat
    for ax in axs[:-1]:
        ax.set_axisbelow(True) # set grid to below
        ax.grid(alpha=0.3)
    axs[-1].axis("off") # set the axis off, this one is for the summary
    plot_age_vs_error(valid_loader, laplace_model, axs[0], lambda x, y: mse(x, y, reduction=None))
    plot_predicted_age_vs_error(valid_loader, laplace_model, axs[1], lambda x, y: mse(x, y, reduction=None), prob=True)
    plot_calibration_laplace(valid_loader, laplace_model, axs[2], lambda x, y: mse(x, y, reduction=None))
    plot_summary(axs[-1], summary)
    plt.tight_layout()
    plt.savefig(args.mdl + "Laplace_Evaluation.jpg")
