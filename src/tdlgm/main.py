from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path

import optuna
import torch
from optuna.exceptions import TrialPruned
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, random_split

from tdlgm.tDLGM import device, tDLGM, tDLGMConfig

DATASET_PATH = Path(__file__).with_name("data").joinpath("shampoo_sales.csv")
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SeriesConfig:
    seq_len: int = 12
    batch_size: int = 8
    epochs: int = 80
    tuning_trials: int = 200
    tuning_epochs: int = 100
    hidden_size: int = 32
    latent_dim: int = 8
    learning_rate: float = 1e-3
    train_fraction: float = 0.8
    seed: int = 42
    tune: bool = False
    shampoo_code: bool = False
    verbose: bool = False
    horizon: int = 10
    artifact_dir: str = "artifacts/tdlgm"
    checkpoint_interval: int = 10


def tdlgm_config(args) -> SeriesConfig:
    return SeriesConfig(
        **{
            k: v
            for k, v in vars(args).items()
            if k in {f.name for f in fields(SeriesConfig)}
        }
    )


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


def checkpoint_payload(model: tDLGM, runtime: SeriesConfig) -> dict[str, object]:
    return {
        "config": asdict(runtime),
        "model_config": asdict(model.config),
        "model_state_dict": model.state_dict(),
    }


def save_checkpoint(model: tDLGM, runtime: SeriesConfig, checkpoint_path: Path) -> Path:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, runtime), checkpoint_path)
    return checkpoint_path


def save_config(runtime: SeriesConfig, model: tDLGM, output_dir: Path, timestamp: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / f"config_{timestamp}.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "timestamp": timestamp,
                "config": asdict(runtime),
                "model_config": asdict(model.config),
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    return config_path


def checkpoint_filename(timestamp: str, epoch: int) -> str:
    return f"checkpoint_{timestamp}_epoch{epoch:04d}.pt"


def load_checkpoint(checkpoint_path: Path) -> tuple[SeriesConfig, tDLGMConfig]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    runtime = SeriesConfig(**checkpoint["config"])
    model_config = tDLGMConfig(**checkpoint["model_config"])
    return runtime, model_config


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(message)s",
    )
    optuna.logging.set_verbosity(
        optuna.logging.INFO  #if verbose else optuna.logging.WARNING, For now I always want to see the optuna logs.
    )


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
    trial: optuna.Trial | None = None,
    save_to: Path | None = None,
) -> tuple[float, float]:
    torch.manual_seed(runtime.seed)

    model, optimizer = build_runtime_model(runtime)
    train_loader, val_loader = make_dataloaders(runtime)
    train_epochs = runtime.epochs if epochs is None else epochs
    checkpoint_interval = max(1, runtime.checkpoint_interval)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    before = evaluate(model, val_loader)
    if runtime.verbose:
        logger.info("Validation loss before training: %.5f", before)

    if save_to is not None:
        config_path = save_config(runtime, model, save_to, timestamp)
        if runtime.verbose:
            logger.info("Saved configuration to %s", config_path)

    model.train()
    for epoch in range(train_epochs):
        epoch_losses = []
        for batch in train_loader:
            x, x_1, y = unpack_batch(batch)
            epoch_losses.append(model.train_step(x, x_1, y, optimizer))

        if runtime.verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            mean_loss = sum(epoch_losses) / max(1, len(epoch_losses))
            logger.info("Epoch %03d: %.5f", epoch + 1, mean_loss)

        if trial is not None:
            val_loss = evaluate(model, val_loader)
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise TrialPruned()

        if save_to is not None and (
            (epoch + 1) % checkpoint_interval == 0 or epoch + 1 == train_epochs
        ):
            checkpoint_path = save_to / checkpoint_filename(timestamp, epoch + 1)
            save_checkpoint(model, runtime, checkpoint_path)
            save_checkpoint(model, runtime, save_to / "checkpoint.pt")
            if runtime.verbose:
                logger.info("Saved checkpoint to %s", checkpoint_path)

    after = evaluate(model, val_loader)
    if not runtime.tuning_trials:
        assert after < before, "Validation loss did not decrease after training"
    if runtime.verbose:
        logger.info("Validation loss after training: %.5f", after)
    return before, after


def tune_hyperparameters(
    base_runtime: SeriesConfig,
) -> SeriesConfig:
    def objective(trial: optuna.Trial) -> float:
        runtime = replace(
            base_runtime,
            seq_len=trial.suggest_categorical("seq_len", [6, 8, 12, 16, 20]),
            batch_size=trial.suggest_categorical("batch_size", [4, 8, 16, 32, 64, 128]),
            hidden_size=trial.suggest_categorical("hidden_size", [16, 32, 64, 128, 256, 512]),
            latent_dim=trial.suggest_categorical("latent_dim", [4, 8, 16, 32, 64, 128]),
            learning_rate=trial.suggest_float(
                "learning_rate",
                1e-5,
                5e-1,
                log=True,
            ),
        )

        _, after = train_model(
            runtime,
            epochs=runtime.tuning_epochs,
            trial=trial,
        )
        return after

    sampler = optuna.samplers.TPESampler(seed=base_runtime.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=base_runtime.tuning_trials)

    best_runtime = replace(base_runtime, **study.best_trial.params)

    logger.info("Best hyperparameters: %s", study.best_trial.params)
    logger.info("Best validation loss during tuning: %.5f", study.best_value)
    return best_runtime


def train(base_runtime) -> Path:
    torch.manual_seed(base_runtime.seed)
    if base_runtime.verbose:
        logger.info(
            "Starting training with %s.", "tuning" if base_runtime.tune else "no tuning"
        )
    runtime = tune_hyperparameters(base_runtime) if base_runtime.tune else base_runtime
    if base_runtime.verbose and not base_runtime.tune:
        logger.info("Skipping hyperparameter tuning.")
    artifact_dir = Path(base_runtime.artifact_dir)
    train_model(runtime, save_to=artifact_dir)
    return artifact_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tDLGM model on a time series dataset."
    )
    parser.add_argument(
        "--shampoo_code",
        type=bool,
        default=False,
        help="Auto-generated code used for a test",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=5,
        help="Context length for the time series dataset",
    )
    parser.add_argument(
        "--horizon", type=int, default=10, help="Horizon for the time series dataset"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=32, help="Hidden size for the tDLGM model"
    )
    parser.add_argument(
        "--latent_dim", type=int, default=8, help="Latent dimension for the tDLGM model"
    )
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=0.8,
        help="Fraction of the dataset to use for training",
    )
    parser.add_argument(
        "--tune", action="store_true", help="Enable hyperparameter tuning with Optuna"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging output"
    )
    parser.add_argument(
        "--artifact_dir",
        type=str,
        default="artifacts/tdlgm",
        help="Directory where the trained checkpoint will be saved",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=10,
        help="Save a checkpoint every N epochs",
    )

    return parser.parse_args()


def setup(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def main() -> None:
    args = parse_args()
    base_runtime = SeriesConfig(**vars(args))

    configure_logging(args.verbose)
    setup(args)
    train(base_runtime)


if __name__ == "__main__":
    main()
