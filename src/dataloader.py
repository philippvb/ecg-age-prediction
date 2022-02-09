from collections.abc import Sequence
import math
import torch
import numpy as np
from torch.utils.data import TensorDataset
import h5py
import numpy as np
import torch
import pandas as pd


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


class GaussianNoiseECGData(TensorDataset):
    """A Class which generates random ECG data from a Gaussian Noise distribution
    """
    def __init__(self, dataset_size, tracings_shape=(12, 4096), seed=123) -> None:
        torch.manual_seed(seed)
        tracings = torch.rand(tuple([dataset_size] + list(tracings_shape)))
        ages = torch.rand((dataset_size, 1))
        super().__init__(tracings, ages)
