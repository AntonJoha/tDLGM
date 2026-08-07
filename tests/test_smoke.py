from dataclasses import replace

import torch

from tdlgm.baseline import Baseline
from tdlgm.baseline import device as baseline_device
from tdlgm.main import build_runtime_model, unpack_batch
from tdlgm.util import SeriesConfig, make_dataloaders


def test_shampoo_dataloaders_return_batches():
    config = SeriesConfig(
        seq_len=5,
        horizon=1,
        batch_size=4,
        shampoo_code=True,
        reduced_dataset=0.2,
    )

    train_loader, val_loader = make_dataloaders(config)

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    x_train, y_train = train_batch
    x_val, y_val = val_batch

    assert x_train.ndim == 2
    assert y_train.ndim == 2
    assert x_val.ndim == 2
    assert y_val.ndim == 2


def test_long_horizon_training_step_works():
    config = SeriesConfig(
        seq_len=5,
        horizon=3,
        batch_size=4,
        shampoo_code=True,
        reduced_dataset=0.2,
    )

    train_loader, _ = make_dataloaders(config)
    x, y = unpack_batch(next(iter(train_loader)))

    tdlgm_model, _ = build_runtime_model(config)
    tdlgm_loss = tdlgm_model.train_step(
        x, y, torch.optim.Adam(tdlgm_model.parameters())
    )
    assert tdlgm_loss >= 0

    baseline_model = Baseline(replace(config, output_dim=config.horizon)).to(
        baseline_device
    )
    baseline_optimizer = torch.optim.Adam(baseline_model.parameters())
    baseline_loss = baseline_model.train_step(
        x.to(baseline_device), y.to(baseline_device), baseline_optimizer
    )
    assert baseline_loss >= 0
