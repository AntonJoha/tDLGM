from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path

import optuna
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, random_split

from tdlgm.tDLGM import device, tDLGM, tDLGMConfig

DATASET_PATH = Path(__file__).with_name("data").joinpath("shampoo_sales.csv")


@dataclass(slots=True)
class SeriesConfig:
    seq_len: int = 12
    batch_size: int = 8
    epochs: int = 80
    tuning_trials: int = 8
    tuning_epochs: int = 20
    hidden_size: int = 32
    latent_dim: int = 8
    learning_rate: float = 1e-3
    train_fraction: float = 0.8
    seed: int = 42


class WindowedSeriesDataset(Dataset):
    def __init__(self, series: torch.Tensor, seq_len: int):
        if series.ndim != 1:
            raise ValueError("series must be 1D")
        if len(series) <= seq_len:
            raise ValueError("series must be longer than seq_len")

        self.series = series
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.series) - self.seq_len

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.series[index : index + self.seq_len + 1]


def load_series(path: Path) -> torch.Tensor:
    values = []

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            values.append(float(row["sales"]))

    series = torch.tensor(values, dtype=torch.float32)
    mean = series.mean()
    std = series.std(unbiased=False).clamp_min(1e-6)
    return (series - mean) / std


def make_dataloaders(config: SeriesConfig) -> tuple[DataLoader, DataLoader]:
    series = load_series(DATASET_PATH)
    dataset = WindowedSeriesDataset(series, config.seq_len)

    train_size = max(1, int(len(dataset) * config.train_fraction))
    val_size = len(dataset) - train_size

    if val_size == 0:
        train_size -= 1
        val_size = 1

    generator = torch.Generator().manual_seed(config.seed)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader


def unpack_batch(
    batch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = batch.to(device)
    x = batch[:, :-1].unsqueeze(-1)
    x_1 = batch[:, 1:].unsqueeze(-1)
    y = batch[:, -1:].unsqueeze(-1)
    return x, x_1, y


def evaluate(model: tDLGM, loader: DataLoader) -> float:
    losses = []
    for batch in loader:
        x, x_1, y = unpack_batch(batch)
        losses.append(model.get_loss(x, x_1, y))
    return sum(losses) / max(1, len(losses))


def build_runtime_model(runtime: SeriesConfig) -> tuple[tDLGM, Adam]:
    model_config = tDLGMConfig(
        input_dim=1,
        hidden_size=runtime.hidden_size,
        latent_dim=runtime.latent_dim,
        output_dim=1,
        layers=2,
        seq_len=runtime.seq_len,
        learning_rate=runtime.learning_rate,
        batch_size=runtime.batch_size,
        seed=runtime.seed,
    )

    model = tDLGM(model_config).to(device)
    optimizer = Adam(model.get_parameters(), lr=runtime.learning_rate)
    return model, optimizer


def train_model(
    runtime: SeriesConfig,
    epochs: int | None = None,
    verbose: bool = True,
) -> tuple[float, float]:
    torch.manual_seed(runtime.seed)

    model, optimizer = build_runtime_model(runtime)
    train_loader, val_loader = make_dataloaders(runtime)
    train_epochs = runtime.epochs if epochs is None else epochs

    before = evaluate(model, val_loader) if verbose else float("nan")
    if verbose:
        print(f"Validation loss before training: {before:.5f}")

    model.train()
    for epoch in range(train_epochs):
        epoch_losses = []
        for batch in train_loader:
            x, x_1, y = unpack_batch(batch)
            epoch_losses.append(model.train_step(x, x_1, y, optimizer))

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            mean_loss = sum(epoch_losses) / max(1, len(epoch_losses))
            print(f"Epoch {epoch + 1:03d}: {mean_loss:.5f}")

    after = evaluate(model, val_loader)
    if verbose:
        print(f"Validation loss after training: {after:.5f}")
    return before, after


def tune_hyperparameters(base_runtime: SeriesConfig) -> SeriesConfig:
    def objective(trial: optuna.Trial) -> float:
        runtime = replace(
            base_runtime,
            seq_len=trial.suggest_categorical("seq_len", [6, 8, 12]),
            batch_size=trial.suggest_categorical("batch_size", [4, 8, 16]),
            hidden_size=trial.suggest_categorical("hidden_size", [16, 32, 64]),
            latent_dim=trial.suggest_categorical("latent_dim", [4, 8, 16]),
            learning_rate=trial.suggest_float(
                "learning_rate",
                1e-4,
                5e-3,
                log=True,
            ),
        )

        _, after = train_model(
            runtime,
            epochs=runtime.tuning_epochs,
            verbose=False,
        )
        return after

    sampler = optuna.samplers.TPESampler(seed=base_runtime.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=base_runtime.tuning_trials)

    best_runtime = replace(base_runtime, **study.best_trial.params)
    print("Best hyperparameters:", study.best_trial.params)
    print(f"Best validation loss during tuning: {study.best_value:.5f}")
    return best_runtime


def train(use_tuning: bool = True) -> None:
    base_runtime = SeriesConfig()
    torch.manual_seed(base_runtime.seed)
    runtime = tune_hyperparameters(base_runtime) if use_tuning else base_runtime
    train_model(runtime)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the tDLGM time-series example.")
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip Optuna hyperparameter tuning and train with the default settings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(use_tuning=not args.no_tune)


if __name__ == "__main__":
    main()
