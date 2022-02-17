
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
from src.dataloader import  BatchDataloader, compute_weights
from laplace import Laplace
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

    print("Building data loaders...")
    # Get csv data
    df = pd.read_csv(config_dict["path_to_csv"], index_col=config_dict["ids_col"])
    ages = df[config_dict["age_col"]]
    # Get h5 data
    f = h5py.File(config_dict["path_to_traces"], 'r')
    traces = f[config_dict["traces_dset"]]
    if config_dict["ids_dset"]:
        h5ids = f[config_dict["ids_dset"]]
        df = df.reindex(h5ids, fill_value=False, copy=True)
    # Train/ val split
    valid_mask = np.arange(len(df)) <= config_dict["n_valid"]
    train_mask = ~valid_mask
    # weights
    weights = compute_weights(ages)
    # Dataloader
    train_loader = BatchDataloader(traces, ages, bs=config_dict["batch_size"], mask=train_mask)
    valid_loader = BatchDataloader(traces, ages, bs=config_dict["batch_size"], mask=valid_mask)

    tqdm.write("Done!")

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
            data = data.to(device).transpose(1,2)
            out_mean, out_var = laplace_model(data)
            var += out_var.cpu().sum()
        var/= len(valid_loader)
        print("Mean var is:", var.item())

        for noise in [0.1, 1, 10]:
            ood_var = torch.tensor([0.0])
            for data, ages in valid_loader:
                data += noise * torch.randn_like(data)
                data = data.to(device).transpose(1,2)
                out_mean, out_var = laplace_model(data)
                ood_var += out_var.cpu().sum()
            ood_var/= len(valid_loader)
            print(f"Mean var with noise of {noise} is:", ood_var.item())

        ood_var = torch.tensor([0.0])
        for data, ages in valid_loader:
            data = torch.flip(data, dims=[-1])
            data = data.to(device).tranpose(1,2)
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
