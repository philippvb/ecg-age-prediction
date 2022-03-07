import torch.nn as nn
import torch
import numpy as np
from src.models import *
from src.loss_functions import mse, gaussian_nll, mae

def _padding(downsample, kernel_size):
    """Compute required padding"""
    padding = max(0, int(np.floor((kernel_size - downsample + 1) / 2)))
    return padding


def _downsample(n_samples_in, n_samples_out):
    """Compute downsample rate"""
    downsample = int(n_samples_in // n_samples_out)
    if downsample < 1:
        raise ValueError("Number of samples should always decrease")
    if n_samples_in % n_samples_out != 0:
        raise ValueError("Number of samples for two consecutive blocks "
                         "should always decrease by an integer factor.")
    return downsample


class ResBlock1d(nn.Module):
    """Residual network unit for unidimensional signals."""

    def __init__(self, n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate):
        if kernel_size % 2 == 0:
            raise ValueError("The current implementation only support odd values for `kernel_size`.")
        super(ResBlock1d, self).__init__()
        # Forward path
        padding = _padding(1, kernel_size)
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(n_filters_out)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        padding = _padding(downsample, kernel_size)
        self.conv2 = nn.Conv1d(n_filters_out, n_filters_out, kernel_size,
                               stride=downsample, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(n_filters_out)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Skip connection
        skip_connection_layers = []
        # Deal with downsampling
        if downsample > 1:
            maxpool = nn.MaxPool1d(downsample, stride=downsample)
            skip_connection_layers += [maxpool]
        # Deal with n_filters dimension increase
        if n_filters_in != n_filters_out:
            conv1x1 = nn.Conv1d(n_filters_in, n_filters_out, 1, bias=False)
            skip_connection_layers += [conv1x1]
        # Build skip conection layer
        if skip_connection_layers:
            self.skip_connection = nn.Sequential(*skip_connection_layers)
        else:
            self.skip_connection = None

    def forward(self, x, y):
        """Residual unit."""
        if self.skip_connection is not None:
            y = self.skip_connection(y)
        else:
            y = y
        # 1st layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # 2nd layer
        x = self.conv2(x)
        x += y  # Sum skip connection and main connection
        y = x
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        return x, y




class ResNet1d(NeuralNetwork):
    """Residual network for unidimensional signals.
    Parameters
    ----------
    input_dim : tuple
        Input dimensions. Tuple containing dimensions for the neural network
        input tensor. Should be like: ``(n_filters, n_samples)``.
    blocks_dim : list of tuples
        Dimensions of residual blocks.  The i-th tuple should contain the dimensions
        of the output (i-1)-th residual block and the input to the i-th residual
        block. Each tuple shoud be like: ``(n_filters, n_samples)``. `n_samples`
        for two consecutive samples should always decrease by an integer factor.
    dropout_rate: float [0, 1), optional
        Dropout rate used in all Dropout layers. Default is 0.8
    kernel_size: int, optional
        Kernel size for convolutional layers. The current implementation
        only supports odd kernel sizes. Default is 17.
    References
    ----------
    .. [1] K. He, X. Zhang, S. Ren, and J. Sun, "Identity Mappings in Deep Residual Networks,"
           arXiv:1603.05027, Mar. 2016. https://arxiv.org/pdf/1603.05027.pdf.
    .. [2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in 2016 IEEE Conference
           on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778. https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, input_dim, blocks_dim, n_classes, kernel_size=17, dropout_rate=0.8):
        super(ResNet1d, self).__init__(loss_name="MSE")
        # First layers
        n_filters_in, n_filters_out = input_dim[0], blocks_dim[0][0]
        n_samples_in, n_samples_out = input_dim[1], blocks_dim[0][1]
        downsample = _downsample(n_samples_in, n_samples_out)
        padding = _padding(downsample, kernel_size)
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, bias=False,
                               stride=downsample, padding=padding)
        self.bn1 = nn.BatchNorm1d(n_filters_out)
        self.relu = nn.ReLU()

        # Residual block layers
        self.res_blocks = []
        for i, (n_filters, n_samples) in enumerate(blocks_dim):
            n_filters_in, n_filters_out = n_filters_out, n_filters
            n_samples_in, n_samples_out = n_samples_out, n_samples
            downsample = _downsample(n_samples_in, n_samples_out)
            resblk1d = ResBlock1d(n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate)
            self.add_module('resblock1d_{0}'.format(i), resblk1d)
            self.res_blocks += [resblk1d]

        # Linear layer
        n_filters_last, n_samples_last = blocks_dim[-1]
        last_layer_dim = n_filters_last * n_samples_last
        self.lin = nn.Linear(last_layer_dim, n_classes)
        self.n_blk = len(blocks_dim)

    def forward(self, x):
        """Implement ResNet1d forward propagation"""
        # First layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Residual blocks
        y = x
        for blk in self.res_blocks:
            x, y = blk(x, y)

        # Flatten array
        x = x.view(x.size(0), -1)

        # Fully conected layer
        x = self.lin(x)
        return x

    def compute_loss(self, traces: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, reduction=torch.sum) -> torch.Tensor:
        # forward pass
        pred_ages = self(traces)
        if torch.is_tensor(weights):
            loss = mse(target, pred_ages, weights=weights, reduction=torch.sum)
        else:
            loss = mse(target, pred_ages, weights=None, reduction=torch.sum)
        return loss

    def evaluate(self, ep, dataload:BatchDataloader, device):
        self.eval()
        n_entries = 0

        eval_desc = "Epoch {:2d}: valid - Loss(WMSE): {:.6f}"
        eval_bar = tqdm(initial=0, leave=True, total=len(dataload),
                        desc=eval_desc.format(ep, 0, 0), position=0)

        # tracking
        total_mse = 0
        total_wmse = 0
        total_wmae = 0

        for traces, ages, weights in dataload:
            traces, ages, weights = traces.to(device), ages.to(device), weights.to(device)

            with torch.no_grad():
                pred_ages = self(traces)
                total_wmse += mse(ages, pred_ages, weights=weights, reduction=torch.sum).cpu().numpy()
                total_mse += mse(ages, pred_ages, weights=None, reduction=torch.sum).cpu().numpy()
                total_wmae += mae(ages, pred_ages, weights, reduction=torch.sum).cpu().numpy()
                # Update outputs
                bs = len(traces)
                n_entries += bs
                # Print result
                eval_bar.desc = eval_desc.format(ep, total_wmse / n_entries)
                eval_bar.update(1)

        eval_bar.close()
        return total_mse / n_entries, total_wmse / n_entries, total_wmae /n_entries




class ProbResNet1d(NeuralNetwork):
    """Residual network for unidimensional signals.
    Parameters
    ----------
    input_dim : tuple
        Input dimensions. Tuple containing dimensions for the neural network
        input tensor. Should be like: ``(n_filters, n_samples)``.
    blocks_dim : list of tuples
        Dimensions of residual blocks.  The i-th tuple should contain the dimensions
        of the output (i-1)-th residual block and the input to the i-th residual
        block. Each tuple shoud be like: ``(n_filters, n_samples)``. `n_samples`
        for two consecutive samples should always decrease by an integer factor.
    dropout_rate: float [0, 1), optional
        Dropout rate used in all Dropout layers. Default is 0.8
    kernel_size: int, optional
        Kernel size for convolutional layers. The current implementation
        only supports odd kernel sizes. Default is 17.
    References
    ----------
    .. [1] K. He, X. Zhang, S. Ren, and J. Sun, "Identity Mappings in Deep Residual Networks,"
           arXiv:1603.05027, Mar. 2016. https://arxiv.org/pdf/1603.05027.pdf.
    .. [2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in 2016 IEEE Conference
           on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778. https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, input_dim, blocks_dim, kernel_size=17, dropout_rate=0.8):
        super(ProbResNet1d, self).__init__(loss_name="NLL")
        # First layers
        n_filters_in, n_filters_out = input_dim[0], blocks_dim[0][0]
        n_samples_in, n_samples_out = input_dim[1], blocks_dim[0][1]
        downsample = _downsample(n_samples_in, n_samples_out)
        padding = _padding(downsample, kernel_size)
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, bias=False,
                               stride=downsample, padding=padding)
        self.bn1 = nn.BatchNorm1d(n_filters_out)
        self.relu = nn.ReLU()

        # Residual block layers
        self.res_blocks = []
        for i, (n_filters, n_samples) in enumerate(blocks_dim):
            n_filters_in, n_filters_out = n_filters_out, n_filters
            n_samples_in, n_samples_out = n_samples_out, n_samples
            downsample = _downsample(n_samples_in, n_samples_out)
            resblk1d = ResBlock1d(n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate)
            self.add_module('resblock1d_{0}'.format(i), resblk1d)
            self.res_blocks += [resblk1d]

        # Linear layer
        self.n_blk = len(blocks_dim)
        n_filters_last, n_samples_last = blocks_dim[-1]
        last_layer_dim = n_filters_last * n_samples_last
        self.lin_mean_1 = nn.Linear(last_layer_dim, last_layer_dim)
        self.lin_mean_2 = nn.Linear(last_layer_dim, 1)
        self.lin_log_var_1 = nn.Linear(last_layer_dim, last_layer_dim)
        self.lin_log_var_2 = nn.Linear(last_layer_dim, 1)#
        self.lin_relu = nn.ReLU()


    def reformat_Laplace(self):
        self.forward = self.forward_mean

    def forward(self, x):
        """Implement ResNet1d forward propagation"""
        x = self.features(x)
        # Fully connected layer
        return self.features_to_mean(x), self.features_to_log_var(x)

    def features_to_mean(self, x):
        return self.lin_mean_2(self.lin_relu(self.lin_mean_1(x)))

    def features_to_log_var(self, x):
        return self.lin_log_var_2(self.lin_relu(self.lin_log_var_1(x)))

    def forward_mean(self, x):
        x = self.features(x)
        return self.features_to_mean(x)

    def forward_log_var(self, x):
        x = self.features(x)
        return self.features_to_log_var(x)

    def features(self, x):
        # First layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Residual blocks
        y = x
        for blk in self.res_blocks:
            x, y = blk(x, y)

        # Flatten array
        x = x.view(x.size(0), -1)
        return x

    def compute_loss(self, traces: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, reduction=torch.sum) -> torch.Tensor:
        # forward pass
        pred_ages, pred_ages_var = self(traces)
        if torch.is_tensor(weights):
            loss, exp, log_var = gaussian_nll(target, pred_ages, pred_ages_var, weights=weights, reduction=torch.sum)
        else:
            loss, exp, log_var = gaussian_nll(target, pred_ages, pred_ages_var, weights=None, reduction=torch.sum)

        return loss      

    def evaluate(self, ep, dataload:BatchDataloader, device):
        """For now same as Normal however change to better method

        Args:
            ep (_type_): _description_
            dataload (BatchDataloader): _description_
            device (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.eval()
        n_entries = 0

        eval_desc = "Epoch {:2d}: valid - Loss(WMSE): {:.6f}"
        eval_bar = tqdm(initial=0, leave=True, total=len(dataload),
                        desc=eval_desc.format(ep, 0, 0), position=0)

        # tracking
        total_mse = 0
        total_wmse = 0
        total_wmae = 0
        total_log_var = 0

        for traces, ages, weights in dataload:
            traces, ages, weights = traces.to(device), ages.to(device), weights.to(device)

            with torch.no_grad():
                pred_ages, pred_log_var = self(traces)
                total_wmse += mse(ages, pred_ages, weights=weights, reduction=torch.sum).cpu().numpy()
                total_mse += mse(ages, pred_ages, weights=None, reduction=torch.sum).cpu().numpy()
                total_wmae += mae(ages, pred_ages, weights, reduction=torch.sum).cpu().numpy()
                total_log_var += pred_log_var.cpu().numpy().sum()
                # Update outputs
                bs = len(traces)
                n_entries += bs
                # Print result
                eval_bar.desc = eval_desc.format(ep, total_wmse / n_entries)
                eval_bar.update(1)

        eval_bar.close()
        return total_mse / n_entries, total_wmse / n_entries, total_wmae /n_entries, total_log_var/n_entries