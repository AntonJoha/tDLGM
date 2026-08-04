from __future__ import annotations

from datetime import datetime, timezone
import json
import argparse
import logging
from dataclasses import asdict, fields, replace
from pathlib import Path

import optuna
import torch
from optuna.exceptions import TrialPruned
from torch.optim import Adam
from torch.utils.data import DataLoader

from tdlgm.tDLGM import device, tDLGM, tDLGMConfig
from tdlgm.util import SeriesConfig, make_dataloaders, save_checkpoint, save_config, checkpoint_filename

logger = logging.getLogger(__name__)


def tdlgm_config(args) -> SeriesConfig:
    return SeriesConfig(
        **{
            k: v
            for k, v in vars(args).items()
            if k in {f.name for f in fields(SeriesConfig)}
        }
    )


def unpack_batch(
    batch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # batch = batch.to(device)
    x, y = batch[0], batch[1]
    if x.ndim == 2:
        x = x.unsqueeze(-1)
    if y.ndim == 2:
        y = y.unsqueeze(-1)

    if y.size(1) != 1:
        raise ValueError(f"tDLGM currently supports horizon=1; got {y.size(1)}")
    x_1 = torch.cat(
        [
            x,
            y,
        ],
        dim=1,
    )[:, 1 : (1 + x.shape[1]), :]
    return x.to(device), x_1.to(device), y.to(device)


def evaluate(model: tDLGM, loader: DataLoader) -> float:
    model.eval()
    losses = []
    for batch in loader:
        x, x_1, y = unpack_batch(batch)
        loss = model.get_loss(x, x_1, y)
        losses.append(float(loss))
    return sum(losses) / max(1, len(losses))

def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(message)s",
    )
    optuna.logging.set_verbosity(
        optuna.logging.INFO  # if verbose else optuna.logging.WARNING, For now I always want to see the optuna logs.
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

    # Save the configuration to a JSON file in the save_to directory
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

        if runtime.verbose:
            mean_loss = sum(epoch_losses) / max(1, len(epoch_losses))
            logger.info("Epoch %03d: %.5f", epoch + 1, mean_loss)


        # optuna trial reporting and pruning
        if trial is not None:
            val_loss = evaluate(model, val_loader)
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise TrialPruned()
        
        # Save checkpoint if save_to is specified and it's time to save
        if save_to is not None and (
            (epoch + 1) % checkpoint_interval == 0 or epoch + 1 == train_epochs
        ):
            checkpoint_path = save_to / checkpoint_filename( f"{(epoch + 1):04d}")
            save_checkpoint(model, runtime, checkpoint_path)
            if runtime.verbose:
                logger.info("Saved checkpoint to %s", checkpoint_path)

    after = evaluate(model, val_loader)
    if runtime.verbose:
        logger.info("Validation loss after training: %.5f", after)
    if trial is None and after >= before:
        logger.warning(
            "Validation loss did not improve: before=%.5f after=%.5f",
            before,
            after,
        )

    # Save final checkpoint if save_to is specified
    if save_to is not None:
        print("Save to: ", save_to)
        check = save_to / checkpoint_filename("final")
        checkpoint_path = save_checkpoint(model, runtime, check)
        if runtime.verbose:
            logger.info("Saved checkpoint to %s", checkpoint_path)
    return before, after


def tune_hyperparameters(
    base_runtime: SeriesConfig,
) -> SeriesConfig:
    def objective(trial: optuna.Trial) -> float:
        runtime = replace(
            base_runtime,
            seq_len=trial.suggest_categorical("seq_len", [6, 8, 12, 16, 20]),
            batch_size=trial.suggest_categorical("batch_size", [4, 8, 16, 32, 64, 128]),
            hidden_size=trial.suggest_categorical(
                "hidden_size", [16, 32, 64, 128, 256, 512]
            ),
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
    study = optuna.create_study(
        direction="minimize", sampler=sampler, pruner=optuna.pruners.MedianPruner()
    )
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
        action="store_true",
        help="Use the bundled shampoo sales dataset.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=12,
        help="Context length for the time series dataset",
    )
    parser.add_argument(
        "--horizon", type=int, default=1, help="Horizon for the time series dataset"
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
        "--baseline",
        action="store_true",
        help="Train a baseline model instead of tDLGM.",
    )
    parser.add_argument(
        "--artifact_dir",
        type=str,
        default="artifacts/tdlgm",
        help="Directory to save checkpoints.",
    )
    parser.add_argument(
        "--reduced_dataset",
        type=float,
        default=None,
        help="Fraction of the dataset to use for training",)
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

    if args.baseline:
        from tdlgm.baseline import baseline_train

        baseline_train(args)
        return

    train(base_runtime)


if __name__ == "__main__":
    main()
