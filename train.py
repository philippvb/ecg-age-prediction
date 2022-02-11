import json
import torch
import os
from tqdm import tqdm
from src.resnet import ResNet1d
from src.dataloader import ECGAgeDataset
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import numpy as np
from datetime import datetime
from src.loss_functions import mse, mae
from src.argparser import parse_ecg_args, parse_ecg_json

def train(ep, dataload):
    model.train()
    total_wmse = 0
    total_mse = 0
    total_wmae = 0
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
        loss = mse(ages, pred_ages, weights)
        # Backward pass
        loss.backward()
        # Optimize
        optimizer.step()
        # Update
        bs = len(traces)
        # calculate tracking metrics
        with torch.no_grad():
            total_wmse += loss.detach().cpu().numpy()
            total_mse += mse(ages, pred_ages, weights=None).cpu().numpy()
            total_wmae += mae(ages, pred_ages, weights).cpu().numpy()

        n_entries += bs
        # Update train bar
        train_bar.desc = train_desc.format(ep, total_mse / n_entries)
        train_bar.update(1)
    train_bar.close()
    return total_wmse / n_entries, total_mse/n_entries, total_wmae/n_entries


def eval(ep, dataload):
    model.eval()
    total_loss = 0
    n_entries = 0
    eval_desc = "Epoch {:2d}: valid - Loss: {:.6f}"
    eval_bar = tqdm(initial=0, leave=True, total=len(dataload),
                    desc=eval_desc.format(ep, 0, 0), position=0)
    for traces, ages, weights in dataload:
        traces, ages, weights = traces.to(device), ages.to(device), weights.to(device)
        with torch.no_grad():
            # Forward pass
            pred_ages = model(traces)
            loss = mse(ages, pred_ages, weights)
            # Update outputs
            bs = len(traces)
            # Update ids
            total_loss += loss.detach().cpu().numpy()
            n_entries += bs
            # Print result
            eval_bar.desc = eval_desc.format(ep, total_loss / n_entries)
            eval_bar.update(1)
    eval_bar.close()
    return total_loss / n_entries


if __name__ == "__main__":
    import pandas as pd
    import argparse
    from warnings import warn
    args = parse_ecg_json()

    torch.manual_seed(args["seed"])
    print(args)
    # Set device
    device = torch.device('cuda:' + args["gpu_id"] if torch.cuda.is_available() else 'cpu')
    folder = args["folder"] + datetime.now().strftime("%y_%m_%d_%H_%M") + "/" # add date

    # Generate output folder if needed
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Save config file
    with open(os.path.join(folder, 'args.json'), 'w') as f:
        json.dump(args, f, indent='\t')

    tqdm.write("Building data loaders...")
    dataset = ECGAgeDataset(args["path_to_traces"], args["path_to_csv"],
     id_key=args["id_key"], tracings_key=args["tracings_key"],
      size=args["dataset_subset"])
    train_dataset_size = int(len(dataset) * (1 - args["valid_split"]))
    train_set, valid_set = random_split(dataset, [train_dataset_size, len(dataset) - train_dataset_size])
    train_loader = DataLoader(train_set, batch_size=args["batch_size"])
    valid_loader = DataLoader(valid_set, batch_size=args["batch_size"])

    tqdm.write("Done!")

    tqdm.write("Define model...")
    N_LEADS = 12  # the 12 leads
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
        train_wmse, train_mse, train_wmae = train(ep, train_loader)
        valid_loss = eval(ep, valid_loader)
        # Save best model
        if valid_loss < best_loss:
            # Save model
            torch.save({'epoch': ep,
                        'model': model.state_dict(),
                        'valid_loss': valid_loss,
                        'optimizer': optimizer.state_dict()},
                       os.path.join(folder, 'model.pth'))
            # Update best validation loss
            best_loss = valid_loss
        # Get learning rate
        for param_group in optimizer.param_groups:
            learning_rate = param_group["lr"]
        # Interrupt for minimum learning rate
        if learning_rate < args["min_lr"]:
            break
        # Print message
        tqdm.write('Epoch {:2d}: \tTrain Loss {:.6f} ' \
                  '\tValid Loss {:.6f} \tLearning Rate {:.7f}\t'
                 .format(ep, train_wmse, valid_loss, learning_rate))
        # Save history
        history = history.append({"epoch": ep, "train_loss": train_wmse,
                                  "valid_loss": valid_loss, "lr": learning_rate,
                                  "weighted_rmse": np.sqrt(train_wmse), "weighted_mae": train_wmae, "rmse": np.sqrt(train_mse),"mse": train_mse},
                                   ignore_index=True)
        history.to_csv(os.path.join(folder, 'history.csv'), index=False)
        # Update learning rate
        scheduler.step(valid_loss)
    tqdm.write("Done!")


