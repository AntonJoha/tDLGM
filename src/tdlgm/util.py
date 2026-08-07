import csv
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from distutils.util import strtobool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

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
    artifact_dir: str = "artifacts_dev/tdlgm"
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
    tdlgm_layers: int = 2

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


# Converts the contents in a .tsf file into a dataframe and returns it along with other meta-data of the dataset: frequency, horizon, whether the dataset contains missing values and whether the series have equal lengths
#
# Parameters
# full_file_path_and_name - complete .tsf file path
# replace_missing_vals_with - a term to indicate the missing values in series in the returning dataframe
# value_column_name - Any name that is preferred to have as the name of the column containing series values in the returning dataframe
def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise ValueError("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise ValueError("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise ValueError(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise ValueError(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise ValueError("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise ValueError("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise ValueError(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise ValueError(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                ).replace(tzinfo=timezone.utc)
                            else:
                                raise ValueError(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise ValueError("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise ValueError("Empty file.")
        if len(col_names) == 0:
            raise ValueError("Missing attribute section.")
        if not found_data_section:
            raise ValueError("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )


def checkpoint_payload(model: nn.Module, runtime: SeriesConfig) -> dict[str, object]:
    return {
        "config": asdict(runtime),
        "model_config": asdict(model.config),
        "model_state_dict": model.state_dict(),
    }


def save_checkpoint(
    model: nn.Module, runtime: SeriesConfig, checkpoint_path: Path
) -> Path:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    saved_path = checkpoint_path.with_name(
        f"{checkpoint_path.name}_{runtime.run_id}.pt"
    )
    torch.save(checkpoint_payload(model, runtime), saved_path)
    return saved_path


def save_config(
    runtime: SeriesConfig, model: nn.Module, output_dir: Path, run_id: str
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / f"config_{run_id}.json"
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


def checkpoint_filename(suffix: str) -> str:
    return f"checkpoint_epoch{suffix}"


def load_checkpoint(
    checkpoint_path: Path,
) -> tuple[SeriesConfig, SeriesConfig, dict[str, torch.Tensor]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    runtime = SeriesConfig(**checkpoint["config"])
    model_config = SeriesConfig(**checkpoint["model_config"])
    model_state = checkpoint["model_state_dict"]
    return runtime, model_config, model_state


class TimeSeriesDataset(Dataset):
    def __init__(self, series, context_length, horizon, mean, std):
        self.series = torch.tensor(series, dtype=torch.float32)
        self.context_length = context_length
        self.horizon = horizon

        # normalize the series
        # Compute statistics from the dataset
        self.mean = mean
        self.std = std

    def __len__(self):
        return max(0, len(self.series) - self.context_length - self.horizon + 1)

    def __getitem__(self, idx):
        x = (self.series[idx : idx + self.context_length] - self.mean) / self.std

        y = (
            self.series[
                idx + self.context_length : idx + self.context_length + self.horizon
            ]
            - self.mean
        ) / self.std

        return x, y


def get_dataset(
    tsf_file_path,
    context_length,
    horizon,
    batch_size=32,
    train_test_split=0.8,
    reduced_dataset=None,
):
    tsf_file_path = Path(tsf_file_path)
    if not tsf_file_path.exists():
        raise FileNotFoundError(tsf_file_path)
    df, _freq, _horizon, _has_missing, _equal_length = convert_tsf_to_dataframe(
        tsf_file_path
    )

    ## normalize data
    all_values = []
    for row in df["series_value"]:
        all_values.extend(row)
    all_values = np.array(all_values)
    mean = np.mean(all_values)
    std = np.std(all_values)

    dataset = None
    for row in df["series_value"]:
        if dataset is None:
            dataset = TimeSeriesDataset(
                row, context_length=context_length, horizon=horizon, mean=mean, std=std
            )
        else:
            dataset = ConcatDataset(
                [
                    dataset,
                    TimeSeriesDataset(
                        row,
                        context_length=context_length,
                        horizon=horizon,
                        mean=mean,
                        std=std,
                    ),
                ]
            )

    if len(dataset) < 2:
        raise ValueError("dataset must contain at least two windows")

    train_size = int(len(dataset) * train_test_split)
    train_size = min(max(1, train_size), len(dataset) - 1)
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
    )

    if reduced_dataset is not None:
        train_size = max(1, int(reduced_dataset * len(train_dataset)))
        test_size = max(1, int(reduced_dataset * len(test_dataset)))
        train_dataset, _ = random_split(
            train_dataset,
            [train_size, len(train_dataset) - train_size],
        )
        test_dataset, _ = random_split(
            test_dataset,
            [test_size, len(test_dataset) - test_size],
        )

    train_df = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_df = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")
    return train_df, test_df


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
    dataset_path = Path(get_dataset_names()[0])

    if config.shampoo_code or not dataset_path.exists():
        if not dataset_path.exists() and not config.shampoo_code:
            logger.warning(
                "Dataset %s not found; falling back to the bundled shampoo data.",
                dataset_path,
            )
        return get_shampoo_dataloaders(config)

    return get_dataset(
        dataset_path,
        config.seq_len,
        config.horizon,
        config.batch_size,
        train_test_split=config.train_fraction,
        reduced_dataset=config.reduced_dataset,
    )


def get_dataset_names():
    return ["data/pedestrian_counts_dataset.tsf"]
