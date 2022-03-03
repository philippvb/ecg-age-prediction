import torch
from tqdm import tqdm
from src.loss_functions import mse
from src.train_loop import train_loop

def train(model, ep, dataload, optimizer, device, weighted=True):
    model.train()
    total_loss = 0
    n_entries = 0
    loss_name = "WMSE" if weighted else "MSE"
    train_desc = "Epoch {:2d}: train - Loss(" + loss_name + "): {:.6f}"
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
        model.zero_grad()
        # Send to device
        # Forward pass
        pred_ages = model(traces)
        if weighted:
            loss = mse(ages, pred_ages, weights=weights, reduction=torch.sum)
        else:
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
    train_loop(train)
