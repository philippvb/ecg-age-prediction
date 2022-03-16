import torch
import numpy as np
import h5py
import numpy as np
import torch
import pandas as pd

MIN_TRAIN_SET_SIZE = 1000 # the minimum size of the dataset before weight computation could become instable


def load_dset_brazilian(args, use_weights=True, map_to_swedish=False, device="cpu"):
    """Loads the dataset given in brazilian format, that is hdf5 for traces and csv for ages. Splits into train and valid

    Args:
        args (dict): The arguments for loading, normally coming from the config.json. Following parameters are needed:
            - path_to_csv: Path to the age csv file
            - ids_col: Name for the (patient) ids column in the csv
            - age_col: Name for the age column in the csv
            - path_to_traces: Path to the traces hdf5 file
            - traces_dset: Name for the traces column 
            - ids_dset: Name for the (patient) ids column in the traces file
            - train_split: How much of the dataset to use for training, on a scale from 0 to 1 (dataset_subset)
            - valid_split: How much of the dataset to use for validation, on a scale from 0 to 1 (n_valid)
        use_weights (bool, optional): Wether to add the weights to the train_loader (always in valid_loader). Defaults to True.
        device (str): The device to push the data to

    Returns:
        tuple(BatchDataloader, BatchDataloader): The train and validation set
    """
    # Get age data in csv
    df = pd.read_csv(args["path_to_csv"], index_col=args["ids_col"])
    ages = df[args["age_col"]]

    # get traces
    f = h5py.File(args["path_to_traces"], 'r')
    traces = f[args["traces_dset"]]

    # define the mapping
    mapping = map_brazilian_to_swedish if map_to_swedish else None
    
    # check dimensions
    if len(ages) != len(traces):
        print("Warning: Length between ages and traces doesn't seem to match")
        if len(ages) < len(traces):
            raise ValueError("Ages csv contains less datapoints than traces")

    # if we have ids in dset, check that indexes match
    if args["ids_dset"]:
        h5ids = f[args["ids_dset"]]
        df = df.reindex(h5ids, fill_value=False, copy=True)

    
    # Train/ val split
    total_length = len(traces)
    valid_mask = np.arange(len(df)) > (total_length * (1 - args["valid_split"]))
    # check total split:
    if args["valid_split"] + args["train_split"] > 1:
        raise ValueError("Sum of train and valid split is larger than 1")
    # take subset if needed, else just take whole remaining
    if args["train_split"] !=1:
        train_mask = np.arange(len(df)) <= args["train_split"] * total_length
    else:
        train_mask = ~valid_mask

    # define dataloader
    weights = compute_weights(ages)
    # for validation we always want weights
    if args["train_split"] * total_length < MIN_TRAIN_SET_SIZE:
        print("Warning: Dataset size seems very small, weights could be inaccurate")
    valid_loader = BatchDataloader(traces, ages, weights, bs=args["batch_size"], mask=valid_mask, transpose=True, traces_mapping=mapping, device=device)
    # train set
    if use_weights:
        train_loader = BatchDataloader(traces, ages, weights, bs=args["batch_size"], mask=train_mask, transpose=True, traces_mapping=mapping, device=device)
    else:
        train_loader = BatchDataloader(traces, ages, bs=args["batch_size"], mask=train_mask, transpose=True, traces_mapping=mapping, device=device)
    return train_loader, valid_loader

def load_dset_swedish(args, use_weights=True, device="cpu"):
    """Loads the dataset given in swedish format, that is hdf5 for both traces and ages. Splits into train and valid

    Args:
        args (dict): The arguments for loading, normally coming from the config.json. Following parameters are needed:
            - path_to_traces: Path to the traces hdf5 file
            - age_col: Name for the age column 
            - traces_dset: Name for the traces column
            - train_split: How much of the dataset to use for training, on a scale from 0 to 1 (dataset_subset)
            - valid_split: How much of the dataset to use for validation, on a scale from 0 to 1 (n_valid)
        use_weights (bool, optional): Wether to add the weights to the dataset. Defaults to True.
        device (str): The device to push the data to

    Returns:
        tuple(BatchDataloader, BatchDataloader): The train and validation set
    """
    f = h5py.File(args["path_to_traces"], 'r')
    traces = f[args["traces_dset"]]
    ages = f[args["age_col"]]
    n_datapoints = len(traces)
    
    # Train/ val split
    if args["valid_split"] + args["train_split"] > 1:
        raise ValueError("Sum of train and valid split is larger than 1")

    valid_mask = np.arange(n_datapoints) > (n_datapoints * (1 - args["valid_split"]))
    # take subset if needed, else just take whole remaining
    if args["train_split"] !=1:
        train_mask = np.arange(n_datapoints) <= args["train_split"] * n_datapoints
    else:
        train_mask = ~valid_mask

    # for valid we always want weights
    if args["train_split"] * n_datapoints < MIN_TRAIN_SET_SIZE:
        print("Warning: Dataset size seems very small, weights could be inaccurate")
    weights = compute_weights(ages)
    valid_loader = BatchDataloader(traces, ages, weights, bs=args["batch_size"], mask=valid_mask, device=device)
    # train loader
    if use_weights:
        train_loader = BatchDataloader(traces, ages, weights, bs=args["batch_size"], mask=train_mask, device=device)
    else:
        train_loader = BatchDataloader(traces, ages, bs=args["batch_size"], mask=train_mask, device=device)

    return train_loader, valid_loader
        

def compute_weights(ages, max_weight=np.inf):
    _, inverse, counts = np.unique(ages, return_inverse=True, return_counts=True)
    weights = 1 / counts[inverse]
    normalized_weights = weights / sum(weights)
    w = len(ages) * normalized_weights
    # Truncate weights to a maximum
    if max_weight < np.inf:
        w = np.minimum(w, max_weight)
        w = len(ages) * w / sum(w)
    return w

def map_brazilian_to_swedish(traces:torch.Tensor) -> torch.Tensor:
    """Maps the ecg traces from the brazilian to swedish format

    Args:
        traces (torch.Tensor): Traces in shape n_datapoints x timesteps x 12 (leads)

    Returns:
        torch.Tensor: reformatted Traces in shape n_datapoints x timesteps x 8 (leads)
    """
    # we have the leads in the following order:
    # DI, DII, DIII, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
    # and we want to drop III, aVR, aVL, aVF, so drop cols 3-5
    indices = torch.tensor([0, 1, 6, 7, 8, 9, 10, 11])
    # TODO: maybe we need to reorder columns
    return torch.index_select(traces, -1, indices)


class BatchDataloader:
    def __init__(self, *tensors, bs=1, mask=None, transpose=False, traces_mapping=None, device="cpu"):
        nonzero_idx, = np.nonzero(mask)
        self.transpose = transpose
        self.tensors = tensors
        self.batch_size = bs
        self.mask = mask
        self.traces_mapping = traces_mapping
        self.device = device
        if nonzero_idx.size > 0:
            self.start_idx = min(nonzero_idx)
            self.end_idx = max(nonzero_idx)+1
        else:
            self.start_idx = 0
            self.end_idx = 0

    def __next__(self):
        if self.start == self.end_idx:
            raise StopIteration
        end = min(self.start + self.batch_size, self.end_idx)
        batch_mask = self.mask[self.start:end]
        while sum(batch_mask) == 0:
            self.start = end
            end = min(self.start + self.batch_size, self.end_idx)
            batch_mask = self.mask[self.start:end]
        batch = [np.array(t[self.start:end]) for t in self.tensors]
        self.start = end
        self.sum += sum(batch_mask)
        out_value = [torch.tensor(b[batch_mask], dtype=torch.float32).to(self.device) for b in batch]
        # remove the leads if necessary
        if self.traces_mapping:
            out_value[0] = self.traces_mapping(out_value[0])
        # transpose dims if necessary
        if self.transpose:
            out_value[0] = out_value[0].transpose(1,2)
        return out_value

    def __iter__(self):
        self.start = self.start_idx
        self.sum = 0
        return self

    def __len__(self):
        count = 0
        start = self.start_idx
        while start != self.end_idx:
            end = min(start + self.batch_size, self.end_idx)
            batch_mask = self.mask[start:end]
            if sum(batch_mask) != 0:
                count += 1
            start = end
        return count

    def get_size(self):
        return sum(self.mask)

    def format_Laplace(self):
        """Formats the dataset for Laplace package by adding the traces to the self.dataset field
        """
        self.dataset = self.tensors[0]

