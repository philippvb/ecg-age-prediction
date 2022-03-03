import json
import torch
import os, sys
sys.path.append(os.getcwd())
from tqdm import tqdm
from src.models.resnet import ProbResNet1d
from src.dataloader import load_dset_bianca, load_dset_standard
import torch.optim as optim
import numpy as np
from datetime import datetime
from src.loss_functions import mse, mae, gaussian_nll
import h5py
from src.argparser import parse_ecg_args, parse_ecg_json

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
        train_loader, valid_loader = load_dset_bianca(args, use_weights=args["use_weights"])
    else:
        train_loader, valid_loader = load_dset_standard(args, use_weights=args["use_weights"])

    tqdm.write("Done!")

    tqdm.write("Define model...")
    N_LEADS = args["n_leads"]  # the 12 leads
    N_CLASSES = 1  # just the age
    model = ProbResNet1d(input_dim=(N_LEADS, args["seq_length"]),
                     blocks_dim=list(zip(args["net_filter_size"], args["net_seq_lengh"])),
                     kernel_size=args["kernel_size"],
                     dropout_rate=args["dropout_rate"])
    model.to(device=device)
    tqdm.write("Done!")

    tqdm.write("Define optimizer...")
    model.create_optimizer(lr=args["lr"])
    tqdm.write("Done!")

    tqdm.write("Define scheduler...")
    model.create_scheduler(patience=args["patience"],min_lr=args["lr_factor"] * args["min_lr"],
                                                     factor=args["lr_factor"])
    tqdm.write("Done!")

    tqdm.write("Training...")
    start_epoch = 0
    best_loss = np.Inf
    history = pd.DataFrame(columns=['epoch', 'train_loss', 'valid_loss', 'lr',
                                    'weighted_rmse', 'weighted_mae', 'rmse', 'mse'])
    for ep in range(start_epoch, args["epochs"]):
        # compute train loss and metrics
        # train_loss = train(ep, train_loader, probabilistic=ep>=args["burnin_epochs"])
        train_loss = model.train_epoch(train_loader, ep, device, weighted=args["use_weights"])
        valid_mse, valid_wmse, valid_wmae = model.evaluate(ep, valid_loader, device)
        # Save best model
        if valid_wmse < best_loss:
            # Save model
            model.save(folder)
            # Update best validation loss
            best_loss = valid_wmse
        # Get learning rate
        for param_group in model.optimizer.param_groups:
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
        model.scheduler.step(valid_wmse)
    tqdm.write("Done!")
