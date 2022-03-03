from collections.abc import Sequence
import math
import torch
import numpy as np
from torch.utils.data import TensorDataset
import h5py
import numpy as np
import torch
import pandas as pd

MIN_TRAIN_SET_SIZE = 1000 # the minimum size of the dataset before weight computation could become instable


def load_dset_standard(args, use_weights=True):
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

    Returns:
        tuple(BatchDataloader, BatchDataloader): The train and validation set
    """
    # Get age data in csv
    df = pd.read_csv(args["path_to_csv"], index_col=args["ids_col"])
    ages = df[args["age_col"]]

    # get traces
    f = h5py.File(args["path_to_traces"], 'r')
    traces = f[args["traces_dset"]]
    
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
    valid_loader = BatchDataloader(traces, ages, weights, bs=args["batch_size"], mask=valid_mask, transpose=True)
    if use_weights:
        if args["train_split"] * total_length < MIN_TRAIN_SET_SIZE:
            print("Warning: Dataset size seems very small, weights could be inaccurate")
        train_loader = BatchDataloader(traces, ages, weights, bs=args["batch_size"], mask=train_mask, transpose=True)
    else:
        train_loader = BatchDataloader(traces, ages, bs=args["batch_size"], mask=train_mask, transpose=True)
    return train_loader, valid_loader

def load_dset_bianca(args, use_weights=True):
    """Loads the dataset given in swedish format, that is hdf5 for both traces and ages. Splits into train and valid

    Args:
        args (dict): The arguments for loading, normally coming from the config.json. Following parameters are needed:
            - path_to_traces: Path to the traces hdf5 file
            - age_col: Name for the age column 
            - traces_dset: Name for the traces column
            - train_split: How much of the dataset to use for training, on a scale from 0 to 1 (dataset_subset)
            - valid_split: How much of the dataset to use for validation, on a scale from 0 to 1 (n_valid)
        use_weights (bool, optional): Wether to add the weights to the dataset. Defaults to True.

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

    # Dataloader
    if use_weights:
        if args["train_split"] * n_datapoints < MIN_TRAIN_SET_SIZE:
            print("Warning: Dataset size seems very small, weights could be inaccurate")
        weights = compute_weights(ages)
        train_loader = BatchDataloader(traces, ages, weights, bs=args["batch_size"], mask=train_mask)
        valid_loader = BatchDataloader(traces, ages, weights, bs=args["batch_size"], mask=valid_mask)
    else:
        train_loader = BatchDataloader(traces, ages, bs=args["batch_size"], mask=train_mask)
        valid_loader = BatchDataloader(traces, ages, bs=args["batch_size"], mask=valid_mask)

    return train_loader, valid_loader



class ECGAgeDataset(TensorDataset):
    def __init__(self, tracing_filepath, metadata_filepath, size=1, id_key="exam_id", tracings_key="tracings", add_weights=True) -> None:
        f = h5py.File(tracing_filepath, 'r')
        dataset_crop = int(len(f[id_key]) * size) if size < 1 else -1
        exam_id = np.array(f[id_key])[:dataset_crop]
        traces = f[tracings_key][:dataset_crop]
        traces = torch.from_numpy(np.array(traces)).transpose(1,2)
        df = pd.read_csv(metadata_filepath)
        df = df.set_index(id_key)
        ages = torch.unsqueeze(torch.tensor(df.loc[exam_id]["age"].values), dim=-1)
        weights = torch.unsqueeze(torch.tensor(ECGAgeDataset.compute_weights(ages)), dim=-1)

        if add_weights:
            super().__init__(traces, ages, weights)
        else:
            super().__init__(traces, ages)

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

class ECGAgeBianca(TensorDataset):

    def __init__(self, tracing_filepath, size=1, tracings_key="x_ecg_train_nodup",  ages_key="x_age_train_nodup", add_weights=True) -> None:
        f = h5py.File(tracing_filepath, 'r')
        dataset_crop = int(len(f[tracings_key]) * size) if size < 1 else -1
        traces = f[tracings_key][:dataset_crop]
        traces = torch.from_numpy(np.array(traces)).transpose(1,2)
        ages = torch.tensor(f[ages_key][:dataset_crop])
        weights = torch.unsqueeze(torch.tensor(ECGAgeDataset.compute_weights(ages)), dim=-1)

        if add_weights:
            super().__init__(traces, ages, weights)
        else:
            super().__init__(traces, ages)

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


# OOD datasets
class GaussianNoiseECGData(TensorDataset):
    """A Class which generates random ECG data from a Gaussian Noise distribution
    """
    def __init__(self, dataset_size, tracings_shape=(12, 4096), seed=123) -> None:
        torch.manual_seed(seed)
        tracings = torch.rand(tuple([dataset_size] + list(tracings_shape)))
        ages = torch.rand((dataset_size, 1))
        super().__init__(tracings, ages)

class FlippedECGAge(ECGAgeDataset):
    def __init__(self, tracing_filepath, metadata_filepath, size=1, id_key="exam_id", tracings_key="tracings", add_weights=True) -> None:
        super().__init__(tracing_filepath, metadata_filepath, size, id_key, tracings_key, add_weights)
        self.tensors[0] = torch.flip(self.tensors, dims=[-1])

class NoisyECGAge(ECGAgeDataset):
    def __init__(self, tracing_filepath, metadata_filepath, noise=1, size=1, id_key="exam_id", tracings_key="tracings", add_weights=True) -> None:
        super().__init__(tracing_filepath, metadata_filepath, size, id_key, tracings_key, add_weights)
        self.tensors[0] += noise * torch.randn_like(self.tensors[0])

        

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


class BatchDataloader:
    def __init__(self, *tensors, bs=1, mask=None, transpose=False):
        nonzero_idx, = np.nonzero(mask)
        self.transpose = transpose
        self.tensors = tensors
        self.batch_size = bs
        self.mask = mask
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
        out_value = [torch.tensor(b[batch_mask], dtype=torch.float32) for b in batch]
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
