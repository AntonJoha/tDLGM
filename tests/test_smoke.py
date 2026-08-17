import math
from collections import deque
from dataclasses import replace
from pathlib import Path

import torch
from tdlgm.baseline import Baseline
from tdlgm.baseline import device as baseline_device
from tdlgm.util import SeriesConfig, checkpoint_filename, make_dataloaders
from torch import nn

from tdlgm import TDLGM
from tdlgm import baseline as baseline_module
from tdlgm.main import build_runtime_model, unpack_batch
from tdlgm.main import train_model as tdlgm_train_model


def test_shampoo_dataloaders_return_batches():
    config = SeriesConfig(
        seq_len=5,
        horizon=1,
        batch_size=4,
        shampoo_code=True,
        reduced_dataset=0.2,
    )

    train_loader, val_loader, test_loader = make_dataloaders(config)

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    x_train, y_train = train_batch
    test_batch = next(iter(test_loader))

    x_val, y_val = val_batch
    x_test, y_test = test_batch

    assert x_train.ndim == 2
    assert y_train.ndim == 2
    assert x_val.ndim == 2
    assert y_val.ndim == 2
    assert x_test.ndim == 2
    assert y_test.ndim == 2


def test_core_tdlgm_is_exported_from_package():
    config = SeriesConfig()

    model = TDLGM(config)

    assert model.config == config
    assert any(isinstance(module, nn.MultiheadAttention) for module in model.modules())
    assert not any(isinstance(module, nn.LSTM) for module in model.modules())


def test_long_horizon_training_step_works():
    config = SeriesConfig(
        seq_len=5,
        horizon=3,
        batch_size=4,
        shampoo_code=True,
        reduced_dataset=0.2,
    )

    train_loader, _, _ = make_dataloaders(config)
    x, y = unpack_batch(next(iter(train_loader)))

    tdlgm_model, _ = build_runtime_model(replace(config, output_dim=config.horizon))
    tdlgm_loss = tdlgm_model.train_step(
        x, y, torch.optim.Adam(tdlgm_model.parameters())
    )
    assert isinstance(tdlgm_loss, float)
    assert not math.isnan(tdlgm_loss)

    baseline_model = Baseline(replace(config, output_dim=config.horizon)).to(
        baseline_device
    )
    baseline_optimizer = torch.optim.Adam(baseline_model.parameters())
    baseline_loss = baseline_model.train_step(
        x.to(baseline_device), y.to(baseline_device), baseline_optimizer
    )
    assert isinstance(baseline_loss, float)
    assert not math.isnan(baseline_loss)


class _FakeTrainer(nn.Module):
    def __init__(self, config: SeriesConfig):
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.zeros(()))
        self.train_steps = 0

    def train_step(self, x, y, optimizer):
        self.train_steps += 1
        return 0.0

    def compute_losses(self, x, y, prior=False):
        return 0.0, 0.0, 0.0

    def get_loss(self, x, y):
        return 0.0


def test_tdlgm_early_stopping_saves_best_checkpoint(monkeypatch, tmp_path):
    config = SeriesConfig(
        seq_len=5,
        horizon=1,
        batch_size=2,
        epochs=80,
        checkpoint_interval=999,
        shampoo_code=True,
        reduced_dataset=0.2,
    )
    batch = (torch.zeros(1, 5), torch.zeros(1, 1))
    save_calls: list[Path] = []
    expected_train_steps = config.early_stopping_patience + 1
    expected_eval_calls = config.early_stopping_patience + 3
    # pre-training eval, one improving epoch, patience flat epochs, final eval
    val_losses = deque([1.0, 0.5] + [0.6] * config.early_stopping_patience + [0.6])
    evaluate_calls = {"count": 0}

    fake_model = _FakeTrainer(replace(config, output_dim=config.horizon))
    optimizer = torch.optim.Adam(fake_model.parameters())

    monkeypatch.setattr(
        "tdlgm.main.build_runtime_model",
        lambda runtime: (fake_model, optimizer),
    )
    monkeypatch.setattr(
        "tdlgm.main.make_dataloaders",
        lambda runtime: ([batch], [batch], [batch]),
    )

    def fake_evaluate(model, loader):
        evaluate_calls["count"] += 1
        return val_losses.popleft()

    monkeypatch.setattr("tdlgm.main.evaluate", fake_evaluate)
    monkeypatch.setattr(
        "tdlgm.main.save_checkpoint",
        lambda model, runtime, checkpoint_path: (
            save_calls.append(checkpoint_path) or checkpoint_path.with_suffix(".pt")
        ),
    )

    tdlgm_train_model(config, save_to=tmp_path)

    assert fake_model.train_steps == expected_train_steps
    assert not val_losses
    assert evaluate_calls["count"] == expected_eval_calls
    assert any(path.name.startswith(checkpoint_filename("best")) for path in save_calls)


def test_baseline_early_stopping_saves_best_checkpoint(monkeypatch, tmp_path):
    config = SeriesConfig(
        seq_len=5,
        horizon=1,
        batch_size=2,
        epochs=80,
        checkpoint_interval=999,
        shampoo_code=True,
        reduced_dataset=0.2,
    )
    batch = (torch.zeros(1, 5), torch.zeros(1, 1))
    save_calls: list[Path] = []
    expected_train_steps = config.early_stopping_patience + 1
    expected_eval_calls = config.early_stopping_patience + 3
    # pre-training eval, one improving epoch, patience flat epochs, final eval
    val_losses = deque([1.0, 0.5] + [0.6] * config.early_stopping_patience + [0.6])
    evaluate_calls = {"count": 0}

    fake_model = _FakeTrainer(replace(config, output_dim=config.horizon))

    monkeypatch.setattr(
        baseline_module,
        "Baseline",
        lambda runtime: fake_model,
    )
    monkeypatch.setattr(
        baseline_module,
        "make_dataloaders",
        lambda runtime: ([batch], [batch], [batch]),
    )

    def fake_evaluate(model, loader):
        evaluate_calls["count"] += 1
        return val_losses.popleft()

    monkeypatch.setattr(baseline_module, "evaluate", fake_evaluate)
    monkeypatch.setattr(
        baseline_module,
        "save_checkpoint",
        lambda model, runtime, checkpoint_path: (
            save_calls.append(checkpoint_path) or checkpoint_path.with_suffix(".pt")
        ),
    )

    baseline_module.train_model(config, save_to=tmp_path)

    assert fake_model.train_steps == expected_train_steps
    assert not val_losses
    assert evaluate_calls["count"] == expected_eval_calls
    assert any(path.name.startswith(checkpoint_filename("best")) for path in save_calls)
