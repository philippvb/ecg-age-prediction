
# Imports
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
from src.dataloader import ECGAgeDataset
from laplace import Laplace



if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--mdl', default="./model",
                        help='folder containing model.')
    parser.add_argument('--path_to_traces', default="/home/caran948/datasets/ecg-traces/preprocessed/traces.hdf5",
                        help='path to file containing ECG traces')
    parser.add_argument('--path_to_csv', default="/home/caran948/datasets/ecg-traces/annotations.csv",
                        help='path to csv file containing attributes.')                        
    parser.add_argument('--batch_size', type=int, default=8,
                        help='number of exams per batch.')
    parser.add_argument('--output', type=str, default='predicted_age.csv',
                        help='output file.')
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
    dataset_subset = 0.001
    dataset = ECGAgeDataset(args.path_to_traces, args.path_to_csv, device=device, id_key="id_exam", tracings_key="signal", size=dataset_subset, add_weights=False)
    dataset = torch.utils.data.DataLoader(dataset, batch_size=32)

    print("Estimating Laplace model")
    # add Laplace
    laplace_model = Laplace(model, "regression", subset_of_weights='last_layer', hessian_structure='full')
    laplace_model.fit(dataset)

    print("Estimating hyperparameters")
    log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
    hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)
    for i in tqdm(range(10)):
        hyper_optimizer.zero_grad()
        neg_marglik = - laplace_model.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
        neg_marglik.backward()
        hyper_optimizer.step()

    print(laplace_model.posterior_covariance)
    print(laplace_model.sigma_noise.item())

    # # add Laplace
    # laplace_set =ECGAgeDataset(args.path_to_traces, args.path_to_csv, device=device, id_key="id_exam", tracings_key="signal", size=dataset_subset, add_weights=False)
    # laplace_loader = torch.utils.data.DataLoader(laplace_set, batch_size=args.batch_size)
    # laplace_model = Laplace(model, "regression", subset_of_weights='last_layer', hessian_structure='full')
    # laplace_model.fit(laplace_loader)
