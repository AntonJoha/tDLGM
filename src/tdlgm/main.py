from __future__ import annotations

import argparse

import csv
from dataclasses import dataclass, fields
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, random_split

from tdlgm.tDLGM import device, tDLGM, tDLGMConfig
from tdlgm.util import get_dataset_names, get_dataset


DATASET_PATH = Path(__file__).with_name("data").joinpath("shampoo_sales.csv")


@dataclass(slots=True)
class SeriesConfig:
    seq_len: int = None
    batch_size: int = None
    epochs: int = None
    hidden_size: int = None
    latent_dim: int = None
    learning_rate: float = None
    train_fraction: float = None


def tdlgm_config(args) -> SeriesConfig:
    return SeriesConfig(**{
    k: v
    for k, v in vars(args).items()
    if k in {f.name for f in fields(SeriesConfig)}})

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

    generator = torch.Generator().manual_seed(42)
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


def train_shampoo(args) -> None:

    runtime = tdlgm_config(args)
    model_config = tDLGMConfig(
        input_dim=1,
        hidden_size=runtime.hidden_size,
        latent_dim=runtime.latent_dim,
        output_dim=1,
        layers=2,
        seq_len=runtime.seq_len,
        learning_rate=runtime.learning_rate,
        batch_size=runtime.batch_size,
    )

    model = tDLGM(model_config).to(device)
    optimizer = Adam(model.get_parameters(), lr=runtime.learning_rate)
    train_loader, val_loader = make_dataloaders(runtime)

    before = evaluate(model, val_loader)
    print(f"Validation loss before training: {before:.5f}")

    model.train()
    for epoch in range(runtime.epochs):
        epoch_losses = []
        for batch in train_loader:
            x, x_1, y = unpack_batch(batch)
            epoch_losses.append(model.train_step(x, x_1, y, optimizer))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            mean_loss = sum(epoch_losses) / max(1, len(epoch_losses))
            print(f"Epoch {epoch + 1:03d}: {mean_loss:.5f}")

    after = evaluate(model, val_loader)
    print(f"Validation loss after training: {after:.5f}")
    assert after < before, "Validation loss did not decrease after training"


def train(args) -> None:
    dataset = get_dataset(get_dataset_names()[0], context_length=args.seq_len, horizon=args.horizon)

    



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tDLGM model on a time series dataset.")
    parser.add_argument("--shampoo_code", type=bool, default=False, help="Auto-generated code used for a test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--seq_len", type=int, default=5, help="Context length for the time series dataset")
    parser.add_argument("--horizon", type=int, default=10, help="Horizon for the time series dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument("--hidden_size", type=int, default=32, help="Hidden size for the tDLGM model")
    parser.add_argument("--latent_dim", type=int, default=8, help="Latent dimension for the tDLGM model")
    parser.add_argument("--train_fraction", type=float, default=0.8, help="Fraction of the dataset to use for training")



    return parser.parse_args()


def setup(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def main() -> None:

    args = parse_args()

    setup(args)

    if args.shampoo_code:
        train_shampoo(args)
        return
    
    train(args)


if __name__ == "__main__":
    main()
