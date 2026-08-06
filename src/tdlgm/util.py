import csv
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import time
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(message)s",
    )
    import optuna.logging

    optuna.logging.set_verbosity(
        optuna.logging.INFO if verbose else optuna.logging.WARNING
    )




DATASET_PATH = Path(__file__).with_name("data").joinpath("shampoo_sales.csv")


@dataclass(slots=True)
class DataConfig:
    seq_len: int = 12
    batch_size: int = 8
    horizon: int = 1
    shampoo_code: bool = False
    reduced_dataset: float | None = None
    train_fraction: float = 0.8
    artifact_dir: str = "artifacts/tdlgm"
    checkpoint_interval: int = 10
    run_id: str | None = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    model_name: str = "tdlgm"


@dataclass(slots=True)
class BaselineConfig(DataConfig):
    input_dim: int = 1
    hidden_dim: int = 20
    latent_dim: int = 5
    output_dim: int = 1
    layers: int = 2

    learning_rate: float = 1e-3

    epochs: int = 80
    seed: int = 42
    device: str | None = None


@dataclass(slots=True)
class SeriesConfig(BaselineConfig):
    # Architecture overrides
    hidden_dim: int = 32
    latent_dim: int = 8

    # Training overrides
    batch_size: int = 64
    learning_rate: float = 1e-3

    beta: float = 1e-3
    alpha: float = 1e-2
    weight_decay: float = 1e-5

    std: float = 0.2

    # Training/tuning
    tuning_trials: int = 100
    tuning_epochs: int = 5
    tune: bool = False

    # Misc
    verbose: bool = False
    baseline: bool = False


def checkpoint_payload(model: nn.Module, runtime: SeriesConfig) -> dict[str, object]:
    return {
        "config": asdict(runtime),
        "model_config": asdict(model.config),
        "model_state_dict": model.state_dict(),
    }


def save_checkpoint(model: nn.Module, runtime: SeriesConfig, checkpoint_path: Path) -> Path:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    print("Saving checkpoint to: ", checkpoint_path)
    torch.save(checkpoint_payload(model, runtime), str(checkpoint_path) + f"_{runtime.run_id}.pt")
    return checkpoint_path


def save_config(
    runtime: SeriesConfig, model: nn.Module, output_dir: Path, run_id: str
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / f"config_{runtime.run_id}.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_id": run_id,
                "config": asdict(runtime),
                "model_config": asdict(model.config),
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    return config_path


def checkpoint_filename( epoch) -> str:
    return f"checkpoint_epoch{epoch}"


def load_checkpoint(checkpoint_path: Path) -> tuple[SeriesConfig, SeriesConfig, nn.Module]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    runtime = SeriesConfig(**checkpoint["config"])
    model_config = SeriesConfig(**checkpoint["model_config"])
    model = checkpoint["model_state_dict"]
    return runtime, model_config, model






class WindowedSeriesDataset(Dataset):
    def __init__(self, series: torch.Tensor, seq_len: int, horizon: int = 1) -> None:
        if series.ndim != 1:
            raise ValueError("series must be 1D")
        if len(series) <= seq_len:
            raise ValueError("series must be longer than seq_len")

        self.series = series
        self.seq_len = seq_len
        self.horizon = horizon

    def __len__(self) -> int:
        return max(0, len(self.series) - self.seq_len - self.horizon + 1)

    def __getitem__(self, index: int) -> torch.Tensor:
        sequence = self.series[index : index + self.seq_len]
        target = self.series[index + self.seq_len : index + self.seq_len + self.horizon]
        return sequence, target


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


def get_shampoo_dataloaders(config: SeriesConfig) -> tuple[DataLoader, DataLoader]:
    series = load_series(DATASET_PATH)
    dataset = WindowedSeriesDataset(series, config.seq_len, config.horizon)

    if len(dataset) < 2:
        raise ValueError("shampoo dataset is too small for train/validation split")

    train_size = int(len(dataset) * config.train_fraction)
    train_size = min(max(1, train_size), len(dataset) - 1)
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(config.seed)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    if config.reduced_dataset is not None:
        train_size = max(1, int(config.reduced_dataset * len(train_dataset)))
        val_size = max(1, int(config.reduced_dataset * len(val_dataset)))
        train_dataset, _ = random_split(
            train_dataset,
            [train_size, len(train_dataset) - train_size],
            generator=generator,
        )
        val_dataset, _ = random_split(
            val_dataset,
            [val_size, len(val_dataset) - val_size],
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


def make_dataloaders(config: DataConfig) -> tuple[DataLoader, DataLoader]:
    return get_shampoo_dataloaders(config)
