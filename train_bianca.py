import json
import torch
import os
from tqdm import tqdm
from src.resnet import ResNet1d
from src.dataloader import BatchDataloader, compute_weights
import torch.optim as optim
import numpy as np
from datetime import datetime
from src.loss_functions import mse, mae
from src.argparser import parse_ecg_args, parse_ecg_json
import h5py
from src.evaluation import eval

def train(ep, dataload):
    model.train()
    total_loss = 0
    n_entries = 0
    train_desc = "Epoch {:2d}: train - Loss: {:.6f}"
    train_bar = tqdm(initial=0, leave=True, total=len(dataload),
                     desc=train_desc.format(ep, 0, 0), position=0)
    for traces, ages, weights in dataload:
        traces, ages, weights = traces.to(device), ages.to(device), weights.to(device)
        # Reinitialize grad
        model.zero_grad()
        # Send to device
        # Forward pass
        pred_ages = model(traces)
        loss = mse(ages, pred_ages, weights=None, reduction=torch.sum)
        # Backward pass
        loss.backward()
        # Optimize
        optimizer.step()
        # Update
        bs = len(traces)
        # calculate tracking metrics
        with torch.no_grad():
            total_loss += loss.detach().cpu().numpy()

        n_entries += bs
        # Update train bar
        train_bar.desc = train_desc.format(ep, total_loss / n_entries)
        train_bar.update(1)
    train_bar.close()
    return total_loss / n_entries




if __name__ == "__main__":
    import pandas as pd
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
    # Get h5 data
    f = h5py.File(args["path_to_traces"], 'r')
    traces = f[args["traces_dset"]]
    ages = f[args["age_col"]]
    n_datapoints = len(traces)
    
    # Train/ val split
    valid_mask = np.arange(n_datapoints) <= args["n_valid"]
    # take subset if wanted
    if args["dataset_subset"] !=1:
        train_mask = np.arange(n_datapoints) <= args["dataset_subset"] * len(traces)
    # weights, TODO: compute only for smaller train set
    weights = compute_weights(ages)
    # Dataloader
    train_loader = BatchDataloader(traces, ages, weights, bs=args["batch_size"], mask=train_mask)
    valid_loader = BatchDataloader(traces, ages, weights, bs=args["batch_size"], mask=valid_mask)

    tqdm.write("Done!")

    tqdm.write("Define model...")
    N_LEADS = 8  # the 12 leads
    N_CLASSES = 1  # just the age
    model = ResNet1d(input_dim=(N_LEADS, args["seq_length"]),
                     blocks_dim=list(zip(args["net_filter_size"], args["net_seq_lengh"])),
                     n_classes=N_CLASSES,
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
        train_loss = train(ep, train_loader)
        valid_mse, valid_wmse, valid_wmae = eval(model, ep, valid_loader, device)
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


