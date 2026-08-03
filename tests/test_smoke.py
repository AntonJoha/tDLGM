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
