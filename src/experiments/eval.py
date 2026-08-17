from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data.data import make_dataloaders
from experiments.baseline import Baseline
from experiments.main import unpack_batch
from experiments.util import SeriesConfig, configure_logging, load_checkpoint
from tdlgm import TDLGM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)

nll_loss = nn.GaussianNLLLoss(reduction="none")
mse_loss = nn.MSELoss(reduction="none")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved tDLGM checkpoint on the validation split."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path("artifacts/tdlgm/checkpoint_epochfinal_20260807-105536.pt"),
        help="Path to a saved checkpoint",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging output"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("artifacts/eval"),
        help="Directory to save evaluation results",
    )
    return parser.parse_args()


def nll_position(
    mean: torch.Tensor, y: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:

    if mean.ndim == 2:
        mean = mean.unsqueeze(-1)
    if logvar.ndim == 2:
        logvar = logvar.unsqueeze(-1)
    if y.ndim == 2:
        y = y.unsqueeze(-1)
    m = mean
    v = logvar.exp()

    loss = nll_loss(m, y, v).mean(dim=(0, 2))
    return loss


def mse_position(mean: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if mean.ndim == 2:
        mean = mean.unsqueeze(-1)
    if y.ndim == 2:
        y = y.unsqueeze(-1)
    m = mean
    loss = mse_loss(m, y).mean(dim=(0, 2))
    return loss


@torch.no_grad()
def evaluate_baseline(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    losses = []
    mse_losses = []
    losses_position = []
    mse_losses_position = []
    xs, means, logvars, ys = [], [], [], []

    for x, y in loader:
        mean, logvar = model(x)
        losses_position.append(nll_position(mean, y.squeeze(-1), logvar))
        mse_losses.append(float(mse_loss(mean, y.squeeze(-1)).mean()))
        mse_losses_position.append(mse_position(mean, y.squeeze(-1)))

        losses.append(float(model.loss(mean, y.squeeze(-1), logvar.exp())))
        xs.append(x)
        means.append(mean)
        logvars.append(logvar)
        ys.append(y)

    return {
        "x": xs,
        "mean": means,
        "logvar": logvars,
        "y": ys,
        "losses": losses,
        "losses_position": losses_position,
        "loss": sum(losses) / max(1, len(losses)),
        "loss_position": sum(losses_position) / max(1, len(losses_position)),
        "mse_loss": sum(mse_losses) / max(1, len(mse_losses)),
        "mse_losses": mse_losses,
        "mse_losses_position": mse_losses_position,
        "mse_loss_position": sum(mse_losses_position)
        / max(1, len(mse_losses_position)),
    }


@torch.no_grad()
def evaluate_tdlgm(model: nn.Module, loader: DataLoader) -> float:
    logger.info("Evaluating tDLGM model...")

    model.eval()
    losses = []

    mse_losses = []
    mse_losses_position = []
    losses_position = []
    xs, means, logvars, ys = [], [], [], []
    for batch in loader:
        x, y = unpack_batch(batch)
        mean, logvar, *_ = model(x)

        losses.append(float(model.nllLoss(mean, y.squeeze(-1), logvar.exp())))
        losses_position.append(nll_position(mean, y.squeeze(-1), logvar))

        mse_losses.append(float(mse_loss(mean, y.squeeze(-1)).mean()))
        mse_losses_position.append(mse_position(mean, y.squeeze(-1)))
        xs.append(x)
        means.append(mean)
        logvars.append(logvar)
        ys.append(y)

    return {
        "x": xs,
        "mean": means,
        "logvar": logvars,
        "y": ys,
        "losses": losses,
        "losses_position": losses_position,
        "loss": sum(losses) / max(1, len(losses)),
        "loss_position": sum(losses_position) / max(1, len(losses_position)),
        "mse_loss": sum(mse_losses) / max(1, len(mse_losses)),
        "mse_losses": mse_losses,
        "mse_losses_position": mse_losses_position,
        "mse_loss_position": sum(mse_losses_position)
        / max(1, len(mse_losses_position)),
    }


def remove_pytorch(results: dict) -> dict:
    def remove_tensors(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: remove_tensors(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [remove_tensors(v) for v in obj]
        else:
            return obj

    return remove_tensors(results)


def _set_input_output_dim(runtime: SeriesConfig, loader: DataLoader) -> None:
    for x, y in loader:
        runtime.input_dim = x.shape[-1]
        runtime.output_dim = y.shape[-1]
        break


def benchmark_model(args, model_path: Path) -> None:
    runtime, model_config, model_state, _model_class = load_checkpoint(model_path)
    runtime.reduced_dataset = 1  # SHOULD ALWAYS EVALUATE ON WHOLE DATASET BUT FOR NOW WE'LL USE REDUCED DATASET FOR SPEED
    runtime = replace(runtime, output_dim=runtime.horizon)
    runtime = replace(runtime, batch_size=args.batch_size)

    _, _, test_loader = make_dataloaders(runtime)
    _set_input_output_dim(runtime, test_loader)

    res = None
    if runtime.model_name == "tdlgm":
        model = TDLGM(model_config).to(device)
        model.load_state_dict(model_state)
        _, _, test_loader = make_dataloaders(runtime)
        res = evaluate_tdlgm(model, test_loader)
    elif runtime.model_name == "baseline":
        model = Baseline(runtime).to(device)
        model.load_state_dict(model_state)
        _, _, test_loader = make_dataloaders(runtime)
        res = evaluate_baseline(model, test_loader)

    print(
        f"Model: {runtime.model_name}, Checkpoint: {model_path}, Test Loss: {res['loss']:.5f}, NLL Position Loss: {res['loss_position']}, Test MSE Loss: {res['mse_loss']:.5f} MSE Position Loss: {res['mse_loss_position']}"
    )

    return remove_pytorch(res)


def save_results(output_dir: Path, checkpoint_path: Path, results: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"eval_results_{checkpoint_path.stem}.pt"
    results["checkpoint_path"] = str(checkpoint_path)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    logger.info("Saved evaluation results to %s", output_file)


def main() -> None:
    args = parse_args()
    args.batch_size = 1
    configure_logging(args.verbose)
    res = benchmark_model(args, args.checkpoint_path)
    save_results(args.output_dir, args.checkpoint_path, res)


if __name__ == "__main__":
    main()
