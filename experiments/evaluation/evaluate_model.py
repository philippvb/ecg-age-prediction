
# Imports
import sys, os
sys.path.append(os.getcwd())
from src.models.resnet import ResNet1d
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
from src.loss_functions import mse, mae
from src.evaluations.plotting import *



if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--mdl',
                        help='folder containing model.')
    parser.add_argument('--s', type=float,
                        help='size of the val set to use, between 0 and 1')
    args, unk = parser.parse_known_args()
    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    # Get config
    config = os.path.join(args.mdl, 'args.json')
    with open(config, 'r') as f:
        config_dict = json.load(f)
    config_dict["valid_split"] = args.s
    # Get model
    N_LEADS = config_dict["n_leads"]
    model = ResNet1d(input_dim=(N_LEADS, config_dict['seq_length']),
                     blocks_dim=list(zip(config_dict['net_filter_size'], config_dict['net_seq_lengh'])),
                     n_classes=1,
                     kernel_size=config_dict['kernel_size'],
                     dropout_rate=config_dict['dropout_rate'])
    # load model checkpoint
    model.load(args.mdl)
    model = model.to(device)

    print("Loading data")
    # Get traces
    if config_dict["bianca"]:
        train_loader, valid_loader = load_dset_swedish(config_dict, use_weights=False, device=device)
    else:
        train_loader, valid_loader = load_dset_brazilian(config_dict, use_weights=False, device=device, map_to_swedish=config_dict["n_leads"] == 8)
    valid_dataset_size =  valid_loader.get_size()

    print("Starting evaluation")

    train_bar = tqdm(initial=0, leave=True, total=len(valid_loader), position=0)

    with torch.no_grad():
        total_mse = 0
        total_mae = 0
        total_wmse = 0
        for data, ages, weights in valid_loader:
            data = data.to(device)
            ages = ages.to(device)
            prediction = model(data)
            total_mse += mse(ages, prediction).cpu()
            total_wmse += mse(ages, prediction, weights).cpu()
            total_mae += mae(ages, prediction).cpu()
            train_bar.update(1)
        train_bar.close()
        total_mse /= valid_dataset_size
        total_mae /= valid_dataset_size
        total_wmse /= valid_dataset_size
        print(f"MSE on validation set is: {total_mse}")
        print(f"Sqrt of MSE on validation set is: {total_mse.sqrt()}")
        print(f"Weighted MSE on validation set is: {total_wmse}")
        print(f"MAE on validation set is: {total_mae}")

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.flat
    for ax in axs:
        ax.set_axisbelow(True) # set grid to below
        ax.grid(alpha=0.3)
    plot_age_vs_error(valid_loader, model, axs[0], lambda x, y: mse(x, y, reduction=None), prob=False)
    plot_predicted_age_vs_error(valid_loader, model, axs[1], lambda x, y: mse(x, y, reduction=None), prob=False)
    plot_summary(axs[2], {"MSE": total_mse, "WMSE": total_wmse, "MAE": total_mae})
    plt.tight_layout()
    plt.savefig(args.mdl + "Standard_Evaluation.jpg")