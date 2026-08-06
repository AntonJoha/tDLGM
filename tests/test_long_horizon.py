from tdlgm.baseline import train_model as train_baseline
from tdlgm.main_rvae import train_model as train_tdlgm_new
from tdlgm.util import SeriesConfig


def test_tdlgm_new_train_model_supports_multi_step_horizon():
    runtime = SeriesConfig(
        seq_len=5,
        horizon=3,
        batch_size=4,
        reduced_dataset=0.1,
        shampoo_code=True,
        hidden_dim=8,
        latent_dim=4,
        layers=1,
    )

    before, after = train_tdlgm_new(runtime, epochs=0)

    assert isinstance(before, float)
    assert isinstance(after, float)


def test_baseline_train_model_supports_multi_step_horizon():
    runtime = SeriesConfig(
        seq_len=5,
        horizon=3,
        batch_size=4,
        reduced_dataset=0.1,
        shampoo_code=True,
        hidden_dim=8,
        layers=1,
    )

    before, after = train_baseline(runtime, epochs=0)

    assert isinstance(before, float)
    assert isinstance(after, float)
