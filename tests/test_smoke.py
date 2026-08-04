import torch

from tdlgm.main import unpack_batch
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


def test_unpack_batch_supports_multi_step_horizon():
    x = torch.zeros(2, 5, 1)
    y = torch.ones(2, 3, 1)

    batch = (x, y)

    x_out, x_1_out, y_out = unpack_batch(batch)

    assert x_out.shape == (2, 5, 1)
    assert x_1_out.shape == (2, 5, 1)
    assert y_out.shape == (2, 3, 1)
