import torch.nn as nn
import torch
from tqdm import tqdm
import os
from src.dataloader import BatchDataloader



class NeuralNetwork(nn.Module):
    def __init__(self, loss_name) -> None:
        super().__init__()
        self.loss_name = loss_name
        self.optimizer = None

    def create_optimizer(self, optimizer_type=torch.optim.Adam, **optimizer_kwargs):
        self.optimizer = optimizer_type(self.parameters(), **optimizer_kwargs)

    def create_scheduler(self, scheduler_type=torch.optim.lr_scheduler.ReduceLROnPlateau, **scheduler_kwargs):
        if not self.optimizer:
            raise ValueError("Please init an optimizer before initializng an scheduler")
        self.scheduler = scheduler_type(self.optimizer, **scheduler_kwargs)

    def compute_loss(self, traces:torch.Tensor, target:torch.Tensor, weights:torch.Tensor, reduction=torch.sum) -> torch.Tensor:
        raise NotImplementedError

    def foward(self, x:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def train_epoch(self, dataload:BatchDataloader, ep, device, weighted=True):
        if not self.optimizer:
            raise ValueError("Seems like the optimizer hasn't been created before training")
        self.train()
        total_loss = 0
        n_entries = 0
        weight_des = "weighted " if weighted else ""
        train_desc = "Epoch {:2d}: train - Loss(" + weight_des + self.loss_name + "): {:.6f}"
        train_bar = tqdm(initial=0, leave=True, total=len(dataload),
                        desc=train_desc.format(ep, 0, 0), position=0)

        for batch in dataload:
            if weighted:
                traces, ages, weights = batch
                traces, ages, weights = traces.to(device), ages.to(device), weights.to(device)
            else:
                traces, ages = batch
                traces, ages = traces.to(device), ages.to(device)

            # Reinitialize grad
            self.optimizer.zero_grad()
            # Forward pass
            if weighted:
                loss = self.compute_loss(traces, ages, weights)
            else:
                loss = self.compute_loss(traces, ages, weights=None)
            # Backward pass
            loss.backward()
            # Optimize
            self.optimizer.step()
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

    def evaluate(self, ep, dataload:BatchDataloader, device):
        raise NotImplementedError

    def save(self, path):
        torch.save({'model': self.state_dict(),
            'optimizer': self.optimizer.state_dict()},
            os.path.join(path, 'model.pth'))

    def load(self, path):
        ckpt = torch.load(os.path.join(path, 'model.pth'), map_location=lambda storage, loc: storage)
        self.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])


