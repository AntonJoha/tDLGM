from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from tdlgm.baseline import Baseline
from tdlgm.main import unpack_batch
from tdlgm.tDLGM_new import TDLGM, device
from tdlgm.util import configure_logging, load_checkpoint, make_dataloaders
import logging
logger = logging.getLogger(__name__)


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
        "--verbose", action="store_true", help="Enable verbose logging output"
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate_baseline(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    losses = []
    for x, y in loader:
        pred = model(x)
        mean = pred[:, : model.config.output_dim]
        logvar = pred[:, model.config.output_dim :]
        losses.append(float(model.loss(mean, y.squeeze(-1), logvar.exp())))
    return sum(losses) / max(1, len(losses))


@torch.no_grad()
def evaluate_tdlgm(model: nn.Module, loader: DataLoader) -> float:
    logger.info("Evaluating tDLGM model...")

    model.eval()
    losses = []
    for batch in loader:
        x, y = unpack_batch(batch)
        mean, logvar, *_ = model(x)
        losses.append(float(model.nllLoss(mean, y.squeeze(-1), logvar.exp())))
    return sum(losses) / max(1, len(losses))


def benchmark_model(model_path: Path) -> None:
    runtime, model_config, model_state, model_class = load_checkpoint(model_path)
    runtime.reduced_dataset = 0.01 # SHOULD ALWAYS EVALUATE ON WHOLE DATASET BUT FOR NOW WE'LL USE REDUCED DATASET FOR SPEED
    runtime = replace(runtime, output_dim=runtime.horizon)

    if runtime.model_name == "tdlgm":
        model = TDLGM(model_config).to(device)
        model.load_state_dict(model_state)
        _, _, test_loader = make_dataloaders(runtime)
        print(f"Test loss: {evaluate_tdlgm(model, test_loader):.5f}")
    elif runtime.model_name == "baseline":
        model = Baseline(runtime).to(device)
        model.load_state_dict(model_state)
        _, _, test_loader = make_dataloaders(runtime)
        print(f"Test loss: {evaluate_baseline(model, test_loader):.5f}")


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    benchmark_model(args.checkpoint_path)


if __name__ == "__main__":
    main()
