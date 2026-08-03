import logging
from dataclasses import fields, replace

from datetime import datetime, timezone
import optuna
import torch
from optuna.exceptions import TrialPruned
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path

from tdlgm.util import BaselineConfig, SeriesConfig, make_dataloaders, save_checkpoint, save_config, checkpoint_filename


logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tdlgm_config(args) -> BaselineConfig:
    return BaselineConfig(
        **{
            k: v
            for k, v in vars(args).items()
            if k in {f.name for f in fields(BaselineConfig)}
        }
    )


class Baseline(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.lstm = nn.LSTM(
            config.input_dim,
            config.hidden_size,
            num_layers=config.layers,
            batch_first=True,
        )
        self.linear = nn.Linear(config.hidden_size, config.output_dim * 2)
        self.loss = nn.GaussianNLLLoss()

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x[:, -1, :]

    def train_step(self, x, y, optimizer):
        self.train()
        optimizer.zero_grad()
        pred = self(x)
        mean = pred[:, : self.config.output_dim]
        logvar = pred[:, self.config.output_dim :]
        loss = self.loss(mean, y, logvar.exp())
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def get_loss(self, x, y):

        # Forward pass
        pred = self(x)
        mean = pred[:, : self.config.output_dim]
        logvar = pred[:, self.config.output_dim :]
        # Compute loss
        loss = self.loss(mean, y, logvar.exp())

        return loss.item()


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    losses = []
    for x, y in loader:
        loss = model.get_loss(x, y)
        losses.append(float(loss))
    return sum(losses) / max(1, len(losses))


def train_model(
    runtime: BaselineConfig,
    epochs: int | None = None,
    trial: optuna.Trial | None = None,
    save_to: Path | None = None,
) -> tuple[float, float]:
    torch.manual_seed(runtime.seed)

    model = Baseline(runtime).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=runtime.learning_rate)
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
        for x, y in train_loader:
            epoch_losses.append(model.train_step(x, y, optimizer))

        if runtime.verbose:
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
            seq_len=trial.suggest_categorical("seq_len", [6, 8, 12]),
            hidden_size=trial.suggest_categorical("hidden_size", [16, 32, 64]),
            layers=trial.suggest_categorical("layers", [1, 2, 3, 5, 10]),
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
        direction="minimize",
        sampler=sampler,
    )
    study.optimize(objective, n_trials=base_runtime.tuning_trials)

    best_runtime = replace(base_runtime, **study.best_trial.params)
    if base_runtime.verbose:
        logger.info("Best hyperparameters: %s", study.best_trial.params)
        logger.info("Best validation loss during tuning: %.5f", study.best_value)
    return best_runtime


def baseline_train(args):
    torch.manual_seed(args.seed)
    baseline_config = SeriesConfig(**vars(args))

    runtime = tune_hyperparameters(baseline_config) if args.tune else baseline_config

    artifact_dir = Path(runtime.artifact_dir)
    train_model(runtime, save_to=artifact_dir)
