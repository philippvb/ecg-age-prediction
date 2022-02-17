
# Imports
import sys, os
sys.path.append(os.getcwd())
from src.resnet import ResNet1d
from tqdm import tqdm
import h5py
import torch
import os
import json
import numpy as np
import argparse
from warnings import warn
import pandas as pd
from src.dataloader import  BatchDataloader, compute_weights, ECGAgeDataset
from laplace import Laplace
from torch.utils.data import DataLoader, random_split
from src.plotting import plot_calibration



if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--mdl',
                        help='folder containing model.')
    args, unk = parser.parse_known_args()
    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Get checkpoint
    ckpt = torch.load(os.path.join(args.mdl, 'model.pth'), map_location=lambda storage, loc: storage)
    # Get config
    config = os.path.join(args.mdl, 'args.json')
    with open(config, 'r') as f:
        config_dict = json.load(f)
    # Get model
    N_LEADS = 12
    model = ResNet1d(input_dim=(N_LEADS, config_dict['seq_length']),
                     blocks_dim=list(zip(config_dict['net_filter_size'], config_dict['net_seq_lengh'])),
                     n_classes=1,
                     kernel_size=config_dict['kernel_size'],
                     dropout_rate=config_dict['dropout_rate'])
    # load model checkpoint
    model.load_state_dict(ckpt["model"])
    model = model.to(device)

    print("Loading data")
    # Get traces
    # how much of our dataset we actually use, on a scale from 0 to 1
    dataset = ECGAgeDataset(config_dict["path_to_traces"], config_dict["path_to_csv"],
     id_key=config_dict["ids_col"], tracings_key=config_dict["traces_dset"],
      size=0.01, add_weights=False)
    train_dataset_size = int(len(dataset) * (1 - config_dict["valid_split"]))
    train_set, valid_set = random_split(dataset, [train_dataset_size, len(dataset) - train_dataset_size])
    train_loader = DataLoader(dataset, batch_size=config_dict["batch_size"])
    valid_loader = DataLoader(valid_set, batch_size=config_dict["batch_size"])


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


    with torch.no_grad():
        var = torch.tensor([0.0])
        for data, ages in valid_loader:
            data = data.to(device)
            out_mean, out_var = laplace_model(data)
            var += out_var.cpu().sum()
        var/= len(valid_loader)
        print("Mean var is:", var.item())

        for noise in [0.1, 1, 10]:
            ood_var = torch.tensor([0.0])
            for data, ages in valid_loader:
                data += noise * torch.randn_like(data)
                data = data.to(device)
                out_mean, out_var = laplace_model(data)
                ood_var += out_var.cpu().sum()
            ood_var/= len(valid_loader)
            print(f"Mean var with noise of {noise} is:", ood_var.item())

        ood_var = torch.tensor([0.0])
        for data, ages in valid_loader:
            data = torch.flip(data, dims=[-1])
            data = data.to(device)
            out_mean, out_var = laplace_model(data)
            ood_var += out_var.cpu().sum()
        ood_var/= len(valid_loader)
        print(f"Mean var for flipped is:", ood_var.item())
        

        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, figsize=(10, 10))
        plot_calibration(laplace_model, valid_loader, axs, device=device, data_noise=laplace_model.sigma_noise.item()**2)
        axs.set_title("Calibration")
        axs.set_xlim(0, 2 * var.sqrt())
        axs.set_xlabel("Confidence in form of standard deviation")
        axs.set_ylabel("Absolut error")
        plt.savefig("laplace_calibration.png")
