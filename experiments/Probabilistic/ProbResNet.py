import json
import torch
import os, sys
sys.path.append(os.getcwd())
from tqdm import tqdm
from src.resnet import ProbResNet1d
from src.dataloader import load_dset_bianca, load_dset_standard
import torch.optim as optim
import numpy as np
from datetime import datetime
from src.loss_functions import mse, mae, gaussian_nll
import h5py
from src.argparser import parse_ecg_args, parse_ecg_json
from src.evaluation import eval

def train(ep, dataload, probabilistic=True):
    print("Training with probabilistic", probabilistic)
    model.train()

    # tracking and pbar
    total_loss = 0
    total_exponent = 0
    total_log_var = 0
    n_entries = 0
    train_desc = "Epoch {:2d}: train - Loss: {:.6f} Exp: {:.6f} Log_var {:.6f}"
    train_bar = tqdm(initial=0, leave=True, total=len(dataload),
                     desc=train_desc.format(ep, 0, 0, 0, 0), position=0)

    for batch_idx, (traces, ages, weights) in enumerate(dataload):
        traces = traces.transpose(1,2)
        # Send to device
        traces, ages, weights = traces.to(device), ages.to(device), weights.to(device)
        # Reinitialize grad
        model.zero_grad()

        # Forward pass
        pred_ages, pred_ages_log_var = model(traces)
        if probabilistic:
            loss, exponent, log_var = gaussian_nll(target=ages, pred=pred_ages, pred_log_var=pred_ages_log_var, weights=None)
        else:
            loss = mse(ages, pred_ages, weights=None, reduction=torch.mean)

        # check for nan values
        if torch.isnan(loss):
            raise ValueError("Loss is nan")

        # Backward pass
        loss.backward()
        # Optimize
        optimizer.step()
        # Update
        bs = len(traces)

        # calculate tracking metrics
        with torch.no_grad():
            total_loss += loss.detach().cpu().numpy() * bs # since we took the mean to update we need to scale up again
            if probabilistic:
                total_exponent += exponent.detach().cpu().numpy() * bs
                total_log_var += log_var.detach().cpu().numpy() * bs

        n_entries += bs
        # Update train bar
        train_bar.desc = train_desc.format(ep, total_loss / n_entries, total_exponent/n_entries, total_log_var/n_entries)
        train_bar.update(1)
    train_bar.close()

    return total_loss/ n_entries


if __name__ == "__main__":
    import pandas as pd
    import argparse
    from warnings import warn
    args = parse_ecg_json()

    torch.manual_seed(args["seed"])
    print(args)
    # Set device
    device = torch.device('cuda:' + args["gpu_id"] if torch.cuda.is_available() else 'cpu')
    folder = args["folder"] + args["output_foldername"]

    # Generate output folder if needed
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Save config file
    with open(os.path.join(folder, 'args.json'), 'w') as f:
        json.dump(args, f, indent='\t')

    tqdm.write("Building data loaders...")
    # Get csv data
    if args["bianca"]:
        train_loader, valid_loader = load_dset_bianca(args)
    else:
        train_loader, valid_loader = load_dset_standard(args)

    tqdm.write("Done!")

    tqdm.write("Define model...")
    N_LEADS = 12  # the 12 leads
    N_CLASSES = 1  # just the age
    model = ProbResNet1d(input_dim=(N_LEADS, args["seq_length"]),
                     blocks_dim=list(zip(args["net_filter_size"], args["net_seq_lengh"])),
                     kernel_size=args["kernel_size"],
                     dropout_rate=args["dropout_rate"])
    model.to(device=device)
    tqdm.write("Done!")

    tqdm.write("Define optimizer...")
    optimizer = optim.Adam(model.parameters(), args["lr"])
    tqdm.write("Done!")

    tqdm.write("Define scheduler...")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args["patience"],
                                                     min_lr=args["lr_factor"] * args["min_lr"],
                                                     factor=args["lr_factor"])
    tqdm.write("Done!")

    tqdm.write("Training...")
    start_epoch = 0
    best_loss = np.Inf
    history = pd.DataFrame(columns=['epoch', 'train_loss', 'valid_loss', 'lr',
                                    'weighted_rmse', 'weighted_mae', 'rmse', 'mse'])
    for ep in range(start_epoch, args["epochs"]):
        # compute train loss and metrics
        train_loss = train(ep, train_loader, probabilistic=ep>=args["burnin_epochs"])
        valid_mse, valid_wmse, valid_wmae = eval(model, ep, valid_loader, device, probabilistic=True)
        # Save best model
        if valid_wmse < best_loss:
            # Save model
            torch.save({'epoch': ep,
                        'model': model.state_dict(),
                        'valid_loss': valid_wmse,
                        'optimizer': optimizer.state_dict()},
                       os.path.join(folder, 'model.pth'))
            # Update best validation loss
            best_loss = valid_wmse
        # Get learning rate
        for param_group in optimizer.param_groups:
            learning_rate = param_group["lr"]
        # Interrupt for minimum learning rate
        if learning_rate < args["min_lr"]:
            break
        # Print message
        tqdm.write('Epoch {:2d}: \tTrain Loss {:.6f} ' \
                  '\tValid Loss {:.6f} \tLearning Rate {:.7f}\t'
                 .format(ep, train_loss, valid_wmse, learning_rate))
        # Save history
        history = history.append({"epoch": ep, "train_loss": train_loss,
                                  "valid_loss": valid_wmse, "lr": learning_rate,
                                  "weighted_rmse": np.sqrt(valid_wmse), "weighted_mae": valid_wmae, "rmse": np.sqrt(valid_mse),"mse": valid_mse},
                                   ignore_index=True)
        history.to_csv(os.path.join(folder, 'history.csv'), index=False)
        # Update learning rate
        scheduler.step(valid_wmse)
    tqdm.write("Done!")
