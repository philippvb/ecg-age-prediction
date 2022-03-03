from collections.abc import Sequence
import math
import torch
import numpy as np
from torch.utils.data import TensorDataset
import h5py
import numpy as np
import torch
import pandas as pd


def load_dset_standard(args, use_weights=True):
    # Get csv data
    df = pd.read_csv(args["path_to_csv"], index_col=args["ids_col"])
    ages = df[args["age_col"]]
    f = h5py.File(args["path_to_traces"], 'r')
    traces = f[args["traces_dset"]]
    if args["ids_dset"]:
        h5ids = f[args["ids_dset"]]
        df = df.reindex(h5ids, fill_value=False, copy=True)
    # Train/ val split
    valid_mask = np.arange(len(df)) < args["n_valid"]
    # take subset if wanted
    if args["dataset_subset"] !=1:
        train_mask = np.arange(len(df)) <= args["dataset_subset"] * len(traces)
    else:
        train_mask = ~valid_mask
    # weights, TODO: compute only for smaller train set
    weights = compute_weights(ages)
    # Dataloader
    if use_weights:
        train_loader = BatchDataloader(traces, ages, weights, bs=args["batch_size"], mask=train_mask, transpose=True)
        valid_loader = BatchDataloader(traces, ages, weights, bs=args["batch_size"], mask=valid_mask, transpose=True)
    else:
        train_loader = BatchDataloader(traces, ages, bs=args["batch_size"], mask=train_mask, transpose=True)
        valid_loader = BatchDataloader(traces, ages, bs=args["batch_size"], mask=valid_mask, transpose=True)
    return train_loader, valid_loader

def load_dset_bianca(args, use_weights=True):
    f = h5py.File(args["path_to_traces"], 'r')
    traces = f[args["traces_dset"]]
    ages = f[args["age_col"]]
    n_datapoints = len(traces)
    
    # Train/ val split
    valid_mask = np.arange(n_datapoints) <= args["n_valid"]
    # take subset if wanted
    if args["dataset_subset"] !=1:
        train_mask = np.arange(n_datapoints) <= args["dataset_subset"] * len(traces)
    else:
        train_mask = ~valid_mask
    # weights, TODO: compute only for smaller train set
    weights = compute_weights(ages)
    # Dataloader
    if use_weights:
        train_loader = BatchDataloader(traces, ages, weights, bs=args["batch_size"], mask=train_mask)
        valid_loader = BatchDataloader(traces, ages, weights, bs=args["batch_size"], mask=valid_mask)
    else:
        train_loader = BatchDataloader(traces, ages, bs=args["batch_size"], mask=train_mask)
        valid_loader = BatchDataloader(traces, ages, bs=args["batch_size"], mask=valid_mask)
    return train_loader, valid_loader


class BatchTensors(Sequence):
    def __init__(self, *tensors, bs=4, mask=None):
        self.tensors = tensors
        self.l = len(tensors[0])
        self.bs = bs
        if mask is None:
            self.mask = np.ones(self.l, dtype=bool)
        else:
            self.mask = np.array(mask, dtype=bool)

    def __getitem__(self, idx):
        index = np.cumsum(self.mask)
        start = idx * self.bs
        end = min(start + self.bs, self.l)
        if end - start <= 0:
            raise IndexError
        batch_mask = np.where((start <= index) & (index < end), self.mask, False)
        return [torch.from_numpy(np.array(t[batch_mask])).to(torch.float32) for t in self.tensors]

    def __len__(self):
        return math.ceil(sum(self.mask) / self.bs)

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
