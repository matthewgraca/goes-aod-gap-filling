import numpy as np
import torch
from torchmetrics import Metric
from constants import *


def inv_log_transform(x, alpha=1e-5):
    return np.exp(x) - alpha


def log_transform(x, alpha=1e-5):
    return np.log(x+alpha)


def to_2d(x, height, width):
    return x.to_numpy(dtype=np.float32).reshape((1, height, width))


def weighted_mse_loss(output, target, weights):
    return (weights * ((target - output) ** 2)).sum()


class WeightedMSELoss(Metric):
    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("total", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("n_valid", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, yhat: torch.Tensor, y: torch.Tensor, weights: torch.Tensor):
        assert yhat.shape == y.shape == weights.shape
        self.total += (weights * ((y - yhat) ** 2)).sum()
        self.n_valid += (weights != 0).sum()

    def compute(self):
        return self.total / self.n_valid
