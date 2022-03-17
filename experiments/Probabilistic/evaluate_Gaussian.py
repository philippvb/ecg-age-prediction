
# Imports
import sys, os
sys.path.append(os.getcwd())
from src.models.resnet import ResNet1d, ProbResNet1d
from tqdm import tqdm
import h5py
import torch
import os
import json
import numpy as np
import argparse
from warnings import warn
import pandas as pd
from src.dataset.dataloader import load_dset_swedish, load_dset_brazilian
from src.plotting import plot_calibration
from src.loss_functions import mse, mae
from src.evaluations.plotting import *



if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--mdl',
                        help='folder containing model.')
    args, unk = parser.parse_known_args()
    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Get config
    config = os.path.join(args.mdl, 'args.json')
    with open(config, 'r') as f:
        config_dict = json.load(f)
    # Get model
    N_LEADS = config_dict["n_leads"]
    model = ProbResNet1d(input_dim=(N_LEADS, config_dict['seq_length']),
                     blocks_dim=list(zip(config_dict['net_filter_size'], config_dict['net_seq_lengh'])),
                     kernel_size=config_dict['kernel_size'],
                     dropout_rate=config_dict['dropout_rate'])
    # load model checkpoint
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

    with torch.no_grad():
    
        total_mse = 0
        total_wmse = 0
        total_mae = 0
        for data, ages, weights in valid_loader:
            data = data.to(device)
            ages = ages.to(device)
            prediction = model(data)[0]
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

        var = torch.tensor([0.0])
        for data, ages, weights in valid_loader:
            data = data.to(device)
            out_mean, out_var = model(data)
            var += out_var.exp().cpu().sum()
        var/= valid_dataset_size
        print("Mean std is:", var.sqrt().item())

        for noise in [0.1, 1, 5, 10]:
            ood_var = torch.tensor([0.0])
            for data, ages, weights in valid_loader:
                data += noise * torch.randn_like(data)
                data = data.to(device)
                out_mean, out_var = model(data)
                ood_var += out_var.exp().cpu().sum()
            ood_var/= valid_dataset_size
            print(f"Mean std with noise of {noise} is:", ood_var.sqrt().item())

        ood_var = torch.tensor([0.0])
        for data, ages, weights in valid_loader:
            data = torch.flip(data, dims=[-1])
            data = data.to(device)
            out_mean, out_var = model(data)
            ood_var += out_var.exp().cpu().sum()
        ood_var/= valid_dataset_size
        print(f"Mean std for flipped is:", ood_var.sqrt().item())
    fig, axs = plt.subplots(2,2, figsize=(10, 10))
    axs = axs.flat
    for ax in axs:
        ax.set_axisbelow(True) # set grid to below
        ax.grid(alpha=0.3)
    plot_age_vs_error(valid_loader, model, axs[0], lambda x, y: mse(x, y, reduction=None))
    plot_predicted_age_vs_error(valid_loader, model, axs[1], lambda x, y: mse(x, y, reduction=None), prob=True)
    plot_calibration(valid_loader, model, axs[2], lambda x, y: mse(x, y, reduction=None))
    plot_summary(axs[23], {"MSE": total_mse, "WMSE": total_wmse, "MAE": total_mae})
    plt.tight_layout()
    plt.savefig(args.mdl + "Gaussian_Evaluation.jpg")