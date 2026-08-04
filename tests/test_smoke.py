import torch

from tdlgm.tDLGM import tDLGM, tDLGMConfig
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


def test_generator_uses_temporal_latents_and_conditional_prior():
    config = tDLGMConfig(
        input_dim=1,
        output_dim=1,
        hidden_size=4,
        latent_dim=2,
        layers=2,
    )
    model = tDLGM(config)
    x = torch.randn(3, 4, 1)
    state = model.model_t(x)
    mean, log_var, posterior_z = model.model_r(x)

    prior_mean, prior_log_var = model.model_g.prior_for_latents(state, posterior_z)
    prediction, prediction_log_var, _ = model.model_g(posterior_z, state)

    assert mean.shape == log_var.shape == posterior_z.shape == (3, 4, 2)
    assert prior_mean.shape == prior_log_var.shape == (3, 4, 2)
    assert prediction.shape == prediction_log_var.shape == (3, 4, 1)
