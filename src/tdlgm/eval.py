from __future__ import annotations

import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tdlgm.util import (
    configure_logging,
    load_checkpoint,
    make_dataloaders,
)
from tdlgm.tDLGM_new import device, TDLGM
from tdlgm.main import unpack_batch
from tdlgm.baseline import Baseline


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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




@torch.no_grad()
def evaluate_baseline(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    losses = []
    input_output_pairs = []
    for x, y in loader:
        pred = model(x)
        mean = pred[:, : model.config.output_dim]
        logvar = pred[:, model.config.output_dim :]
        loss = model.loss(mean, y, logvar.exp())
        losses.append(float(loss))
        input_output_pairs.append(
            {
                "input": x.cpu().numpy().tolist(),
                "prediction": mean.cpu().numpy().tolist(),
                "prediction_variance": logvar.exp().cpu().numpy().tolist(),
                "target": y.cpu().numpy().tolist(),
                "loss": float(loss),
            }
        )
    return sum(losses) / max(1, len(losses)), input_output_pairs

@torch.no_grad()
def evaluate_tdlgm(model: TDLGM, loader: DataLoader) -> float:
    model.eval()
    losses = []
    input_output_pairs = []
    for batch in loader:
        x, _, y = unpack_batch(batch)
        mean, logvar, _, _, _ = model(x)
        loss = model.nllLoss(mean, y[:, 0, :], logvar)
        losses.append(float(loss))
        
        input_output_pairs.append(
            {
                "input": x.cpu().numpy().tolist(),
                "prediction": mean.cpu().numpy().tolist(),
                "prediction_variance": logvar.exp().cpu().numpy().tolist(),
                "target": y.cpu().numpy().tolist(),
                "loss": float(loss),
            }
        )
    return sum(losses) / max(1, len(losses)), input_output_pairs








def benchmark_model(checkpoint_path: Path) -> None:
    runtime, model_config, model_state = load_checkpoint(checkpoint_path)
    runtime.reduced_dataset = 0.2
    print(runtime.shampoo_code)
    
    if runtime.model_name == "tdlgm":
        model = TDLGM(model_config).to(device)
        model.load_state_dict(model_state)
        for i in range(10):
            _, val_loader = make_dataloaders(runtime)
            loss, info = evaluate_tdlgm(model, val_loader)
            print(f"Validation loss: {loss:.5f}")
    elif runtime.model_name == "baseline":
        model = Baseline(runtime).to(device)
        model.load_state_dict(model_state)
        _, val_loader = make_dataloaders(runtime)
        loss, info = evaluate_baseline(model, val_loader)
        print(f"Validation loss: {loss:.5f}")




def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    benchmark_model(args.checkpoint_path)
    

if __name__ == "__main__":
    main()
