from __future__ import annotations
import sys
import argparse
import sys
import csv
from dataclasses import dataclass, fields
import logging
from dataclasses import replace

import optuna
from optuna.exceptions import TrialPruned
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, random_split

from tdlgm.tDLGM import device, tDLGM, tDLGMConfig
from tdlgm.util import make_dataloaders, SeriesConfig


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
    #batch = batch.to(device)
    x, y = batch[0], batch[1]
    if x.ndim == 2:
        x = x.unsqueeze(-1)
    if y.ndim == 2:
        y = y.unsqueeze(-1)

    x_1 = torch.cat(
        [
            x,
            y,
        ],
        dim=1,
    )[:, 1:(1+x.shape[1]), :]
    return x, x_1, y


def evaluate(model: tDLGM, loader: DataLoader) -> float:
    losses = []
    for batch in loader:
        x, x_1, y = unpack_batch(batch)
        loss = model.get_loss(x, x_1, y)
        #print(f"Loss: {loss:.5f}")
        losses.append(loss)
    return sum(losses) / max(1, len(losses))


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(message)s",
    )
    optuna.logging.set_verbosity(
        optuna.logging.INFO if verbose else optuna.logging.WARNING,
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
) -> tuple[float, float]:
    torch.manual_seed(runtime.seed)

    model, optimizer = build_runtime_model(runtime)
    train_loader, val_loader = make_dataloaders(runtime)
    train_epochs = runtime.epochs if epochs is None else epochs
    
    print("HERE")
    before = evaluate(model, val_loader)
    print("THERE")
    if runtime.verbose:
        logger.info("Validation loss before training: %.5f", before)

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

    after = evaluate(model, val_loader)
    logger.log(f"Validation loss after training: {after:.5f}")
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
            seq_len=trial.suggest_categorical("seq_len", [6, 8, 12]),
            batch_size=trial.suggest_categorical("batch_size", [4, 8, 16]),
            hidden_size=trial.suggest_categorical("hidden_size", [16, 32, 64]),
            latent_dim=trial.suggest_categorical("latent_dim", [4, 8, 16]),
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
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=base_runtime.tuning_trials)

    best_runtime = replace(base_runtime, **study.best_trial.params)
    if base_runtime.verbose:
        logger.info("Best hyperparameters: %s", study.best_trial.params)
        logger.info("Best validation loss during tuning: %.5f", study.best_value)
    return best_runtime


def train(base_runtime) -> None:
    torch.manual_seed(base_runtime.seed)
    if base_runtime.verbose:
        logger.info(
            "Starting training with %s.", "tuning" if base_runtime.tune else "no tuning"
        )
    runtime = (
        tune_hyperparameters(base_runtime)
        if base_runtime.tune
        else base_runtime
    )
    if base_runtime.verbose and not base_runtime.tune:
        logger.info("Skipping hyperparameter tuning.")
    train_model(runtime)


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
        default=12,
        help="Context length for the time series dataset",
    )
    parser.add_argument(
        "--horizon", type=int, default=12, help="Horizon for the time series dataset"
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
        type=str,
        default=None,
        help="Train a baseline model instead of tDLGM.",
    )
    parser.add_argument(
            "--reduced_dataset", type=float, default=None, help="Fraction of the dataset to use for training")

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
