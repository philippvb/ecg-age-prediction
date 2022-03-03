
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
from src.dataloader import  BatchDataloader, compute_weights, ECGAgeDataset, load_dset_bianca, load_dset_standard
from laplace import Laplace
from torch.utils.data import DataLoader, random_split
from src.plotting import plot_calibration
from src.loss_functions import mse



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
        train_loader, valid_loader = load_dset_bianca(config_dict, use_weights=False)
    else:
        train_loader, valid_loader = load_dset_standard(config_dict, use_weights=False)
    valid_dataset_size =  valid_loader.get_size()

    with torch.no_grad():
    
        total_loss = 0
        for data, ages, weights in valid_loader:
            data = data.to(device)
            ages = ages.to(device)
            total_loss += mse(ages, model(data)[0]).cpu()
        total_loss /= valid_dataset_size
        print(f"MSE on validation set is: {total_loss}")

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
        
        # TODO: make it work
        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(1, figsize=(10, 10))
        # plot_calibration(model, valid_loader, axs, device=device, data_noise=0, log_scale=True)
        # axs.set_title("Calibration")
        # # ax_lim_std = (var + model.sigma_noise.item()**2).sqrt()
        # # axs.set_xlim(ax_lim_std - 0.3, ax_lim_std + 1)
        # axs.set_xlabel("Confidence in form of standard deviation")
        # axs.set_ylabel("Absolut error")
        # plt.show()
        # plt.savefig("gaussian.png")
