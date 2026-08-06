from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import optuna
import torch
from optuna.exceptions import TrialPruned
from torch.optim import SGD
from torch.utils.data import DataLoader

from tdlgm.rvae import TDLGM, device
from tdlgm.util import (
    SeriesConfig,
    checkpoint_filename,
    configure_logging,
    make_dataloaders,
    save_checkpoint,
    save_config,
)

logger = logging.getLogger(__name__)


def unpack_batch(batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = batch
    if x.ndim == 2:
        x = x.unsqueeze(-1)
    if y.ndim == 2:
        y = y.unsqueeze(-1)
    if y.size(1) != 1:
        raise ValueError(f"tDLGM currently supports horizon=1; got {y.size(1)}")
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model: TDLGM, loader: DataLoader) -> float:
    model.eval()
    losses = []
    for batch in loader:
        x, y = unpack_batch(batch)
        mean, logvar, *_ = model(x)
        losses.append(float(model.nllLoss(mean, y[:, 0, :], logvar)))
    model.train()
    return sum(losses) / max(1, len(losses))


def build_runtime_model(runtime: SeriesConfig) -> tuple[TDLGM, SGD]:
    model = TDLGM(runtime).to(device)
    if runtime.verbose:
        logger.info(
            "Parameters: %s",
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
    optimizer = SGD(model.parameters(), lr=runtime.learning_rate)
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
        recon_loss = kl_loss = consistency = 0.0
        recon_loss_p = kl_loss_p = consistency_p = 0.0

        for batch in train_loader:
            x, y = unpack_batch(batch)
            epoch_losses.append(model.train_step(x, y, optimizer))
            t_recon_loss, t_kl_loss, t_consistency = model.compute_losses(
                x,
                y,
                prior=False,
            )
            t_recon_loss_p, t_kl_loss_p, t_consistency_p = model.compute_losses(
                x,
                y,
                prior=True,
            )
            recon_loss += t_recon_loss
            kl_loss += t_kl_loss
            consistency += t_consistency
            recon_loss_p += t_recon_loss_p
            kl_loss_p += t_kl_loss_p
            consistency_p += t_consistency_p

        if runtime.verbose:
            train_batches = max(1, len(train_loader))
            mean_loss = sum(epoch_losses) / max(1, len(epoch_losses))
            recon_loss /= train_batches
            kl_loss /= train_batches
            consistency /= train_batches
            recon_loss_p /= train_batches
            kl_loss_p /= train_batches
            consistency_p /= train_batches

            logger.info("========== Epoch %03d =========", epoch + 1)
            logger.info(
                " Train loss: %.5f: NLL on val set: %.5f",
                mean_loss,
                evaluate(model, val_loader),
            )
            logger.info(
                " Posterior: NLL %.5f: kl_loss %.5f: Cons_loss %.5f",
                recon_loss,
                kl_loss,
                consistency,
            )
            logger.info(
                " Prior: NLL %.5f: kl_loss %.5f: Cons_loss %.5f",
                recon_loss_p,
                kl_loss_p,
                consistency_p,
            )

        if trial is not None:
            val_loss = evaluate(model, val_loader)
            logger.info(
                "Trial %d: Epoch %03d: Validation loss %.5f",
                trial.number,
                epoch + 1,
                val_loss,
            )
            trial.report(val_loss, epoch)
            if trial.should_prune() or val_loss > before * 2:
                raise TrialPruned()

        if save_to is not None and (
            (epoch + 1) % checkpoint_interval == 0 or epoch + 1 == train_epochs
        ):
            saved_path = save_checkpoint(
                model,
                runtime,
                save_to / checkpoint_filename(f"{epoch + 1:04d}"),
            )
            if runtime.verbose:
                logger.info("Saved checkpoint to %s", saved_path)

    after = evaluate(model, val_loader)
    if runtime.verbose:
        logger.info(
            "Validation loss after training: %.5f and before %.5f", after, before
        )
    if trial is None and after >= before:
        logger.warning(
            "Validation loss did not improve: before=%.5f after=%.5f",
            before,
            after,
        )

    if save_to is not None:
        saved_path = save_checkpoint(
            model,
            runtime,
            save_to / checkpoint_filename("final"),
        )
        if runtime.verbose:
            logger.info("Saved checkpoint to %s", saved_path)
    return before, after


def tune_hyperparameters(base_runtime: SeriesConfig) -> SeriesConfig:
    def objective(trial: optuna.Trial) -> float:
        runtime = replace(
            base_runtime,
            hidden_dim=trial.suggest_categorical("hidden_dim", [16, 32, 64, 128, 256]),
            latent_dim=trial.suggest_categorical("latent_dim", [4, 8, 16, 32, 64]),
            learning_rate=trial.suggest_float(
                "learning_rate",
                1e-5,
                5e-3,
                log=True,
            ),
        )
        _, after = train_model(
            runtime,
            epochs=runtime.tuning_epochs,
            trial=trial,
        )
        return after

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=base_runtime.seed),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=base_runtime.tuning_trials)

    logger.info("Best hyperparameters: %s", study.best_trial.params)
    logger.info("Best validation loss during tuning: %.5f", study.best_value)
    return replace(base_runtime, **study.best_trial.params)


def train(base_runtime: SeriesConfig) -> Path:
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
        description="Train an RVAE-style model on a time series dataset."
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
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=32, help="Hidden size for the model"
    )
    parser.add_argument(
        "--latent_dim", type=int, default=8, help="Latent dimension for the model"
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
        default="artifacts/rvae",
        help="Directory to save checkpoints.",
    )
    parser.add_argument(
        "--reduced_dataset",
        type=float,
        default=None,
        help="Fraction of the dataset to use for training",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=10,
        help="Save a checkpoint every N epochs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = SeriesConfig(**vars(args))

    configure_logging(args.verbose)
    train(runtime)


if __name__ == "__main__":
    main()
