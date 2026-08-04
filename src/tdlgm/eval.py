from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import asdict
import torch

from tdlgm.util import (
    configure_logging,
    evaluate,
    load_checkpoint,
    make_dataloaders,
)
from tdlgm.tDLGM import device, tDLGM
from tdlgm.util import SeriesConfig, tDLGMConfig
import json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved tDLGM checkpoint on the validation split."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path("artifacts/tdlgm/checkpoint.pt"),
        help="Path to a saved checkpoint",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging output"
    )
    return parser.parse_args()







def evaluate_checkpoint(checkpoint_path: Path) -> float:
    runtime, model_config = load_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = tDLGM(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, val_loader = make_dataloaders(runtime)
    loss = evaluate(model, val_loader)
    print(f"Validation loss: {loss:.5f}")
    return loss






def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    evaluate_checkpoint(args.checkpoint_path)


if __name__ == "__main__":
    main()
