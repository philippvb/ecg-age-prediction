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

    # Arguments that will be saved in config file
    parser = argparse.ArgumentParser(add_help=True,
                                     description='Train model to predict rage from the raw ecg tracing.')
    parser.add_argument('--epochs', type=int, default=1,
                        help='maximum number of epochs (default: 70)')
    parser.add_argument('--seed', type=int, default=2,
                        help='random seed for number generator (default: 2)')
    parser.add_argument('--sample_freq', type=int, default=400,
                        help='sample frequency (in Hz) in which all traces will be resampled at (default: 400)')
    parser.add_argument('--seq_length', type=int, default=4096,
                        help='size (in # of samples) for all traces. If needed traces will be zeropadded'
                                    'to fit into the given size. (default: 4096)')
    parser.add_argument('--scale_multiplier', type=int, default=10,
                        help='multiplicative factor used to rescale inputs.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size (default: 32).')
    parser.add_argument('--valid_split', type=float, default=0.05,
                        help='fraction of the data used for validation (default: 0.1).')
    parser.add_argument('--test_split', type=float, default=0.15,
                        help='fraction of the data kept away for testing in a latter stage (default: 0.1).')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument("--patience", type=int, default=7,
                        help='maximum number of epochs without reducing the learning rate (default: 7)')
    parser.add_argument("--min_lr", type=float, default=1e-7,
                        help='minimum learning rate (default: 1e-7)')
    parser.add_argument("--lr_factor", type=float, default=0.1,
                        help='reducing factor for the lr in a plateu (default: 0.1)')
    parser.add_argument('--net_filter_size', type=int, nargs='+', default=[64, 128, 196, 256, 320],
                        help='filter size in resnet layers (default: [64, 128, 196, 256, 320]).')
    parser.add_argument('--net_seq_lengh', type=int, nargs='+', default=[4096, 1024, 256, 64, 16],
                        help='number of samples per resnet layer (default: [4096, 1024, 256, 64, 16]).')
    parser.add_argument('--dropout_rate', type=float, default=0.8,
                        help='dropout rate (default: 0.8).')
    parser.add_argument('--kernel_size', type=int, default=17,
                        help='kernel size in convolutional layers (default: 17).')
    parser.add_argument('--folder', default='model/',
                        help='output folder (default: ./out)')
    parser.add_argument('--traces_dset', default='tracings',
                        help='traces dataset in the hdf5 file.')
    parser.add_argument('--age_col', default='age',
                        help='column with the age in csv file.')
    parser.add_argument('--gpu_id', default='0',
                        help='Which gpu to use.')
    parser.add_argument('--n_valid', type=int, default=100,
                        help='the first `n_valid` exams in the hdf will be for validation.'
                             'The rest is for training') # how is this different from train/valid split above
    parser.add_argument('--path_to_traces', default="/home/caran948/datasets/ecg-traces/preprocessed/traces.hdf5",
                        help='path to file containing ECG traces')
    parser.add_argument('--path_to_csv', default="/home/caran948/datasets/ecg-traces/annotations.csv",
                        help='path to csv file containing attributes.')
    parser.add_argument('--dataset_subset', default=0.001,
                        help='Size of the subset of dataset to take')
    parser.add_argument('--json_config_file', default="/home/phba123/code/ecg-age-prediction/args.json")
    args, unk = parser.parse_known_args()
    args = vars(args)
    if args["json_config_file"]:
        with open(args["json_config_file"], "r") as f:
            default_config = json.load(f)
            default_config.update(args)
            args = default_config

    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")

    torch.manual_seed(args.seed)
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
    dataset = ECGAgeDataset(args["path_to_traces"], args["path_to_csv"], device=device,
     id_key="id_exam", tracings_key="signal",
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


