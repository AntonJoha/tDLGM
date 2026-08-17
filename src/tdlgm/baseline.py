from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

import experiments.baseline as _baseline

device = _baseline.device


class Baseline(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.output_dim == config.horizon and config.input_dim == 1:
            config = replace(config, output_dim=1)
        self.config = config
        self.lstm = nn.LSTM(
            config.input_dim,
            config.hidden_dim,
            num_layers=config.layers,
            batch_first=True,
        )
        self.linear = nn.Linear(
            config.hidden_dim, config.output_dim * 2 * config.horizon
        )
        self.loss = nn.GaussianNLLLoss()

    def _to_output_shape(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), self.config.horizon, self.config.output_dim)
        return x.squeeze(-1) if self.config.output_dim == 1 else x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        x, _ = self.lstm(x)
        x = self.linear(x)[:, -1, :]
        mean = self._to_output_shape(
            x[:, : self.config.output_dim * self.config.horizon]
        )
        logvar = self._to_output_shape(
            x[:, self.config.output_dim * self.config.horizon :]
        )
        return mean, logvar

    def _target(self, y: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        target = y.squeeze(-1)
        if mean.shape != target.shape:
            raise ValueError(
                "prediction and target shapes must match "
                "(output_dim should equal horizon): "
                f"{mean.shape} != {target.shape}"
            )
        return target

    def train_step(
        self, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer
    ) -> float:
        self.train()
        optimizer.zero_grad()
        mean, logvar = self(x)
        loss = self.loss(mean, self._target(y, mean), logvar.exp())
        loss.backward()
        optimizer.step()
        return float(loss)

    @torch.no_grad()
    def get_loss(self, x: torch.Tensor, y: torch.Tensor) -> float:
        mean, logvar = self(x)
        return float(self.loss(mean, self._target(y, mean), logvar.exp()))


evaluate = _baseline.evaluate
make_dataloaders = _baseline.make_dataloaders
save_checkpoint = _baseline.save_checkpoint
save_config = _baseline.save_config
tune_hyperparameters = _baseline.tune_hyperparameters


def _sync_backend() -> None:
    _baseline.Baseline = Baseline
    _baseline.evaluate = evaluate
    _baseline.make_dataloaders = make_dataloaders
    _baseline.save_checkpoint = save_checkpoint
    _baseline.save_config = save_config
    _baseline.tune_hyperparameters = tune_hyperparameters


def train_model(*args, **kwargs):
    _sync_backend()
    return _baseline.train_model(*args, **kwargs)


def baseline_train(*args, **kwargs):
    _sync_backend()
    return _baseline.baseline_train(*args, **kwargs)


__all__ = [
    "Baseline",
    "baseline_train",
    "device",
    "evaluate",
    "make_dataloaders",
    "save_checkpoint",
    "save_config",
    "train_model",
    "tune_hyperparameters",
]
