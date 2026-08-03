from dataclasses import dataclass
from itertools import chain

import argparse

import optuna
import torch
from torch import nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Baseline(nn.Module):
    def __init__(self, config):
        super(Baseline, self).__init__()

        self.config = config
        self.lstm = nn.LSTM(config.input_dim, config.hidden_size, num_layers=config.layers, batch_first=True)
        self.linear = nn.Linear(config.hidden_size, config.output_dim*2)
        self.loss = nn.GaussianNLLLoss()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x


    def train_step(self, x, y, optimizer):
        optimizer.zero_grad()
        pred = self(x)
        mean = pred[:, :, :self.config.output_dim]
        logvar = pred[:, :, self.config.output_dim:]
        loss = self.loss(mean, y, logvar.exp())
        loss.backward()
        optimizer.step()
        return loss.item()

    
    @torch.no_grad()
    def get_loss(self, x, y):

        # Forward pass
        pred = self(x)
        mean = pred[:, :, :self.config.output_dim]
        logvar = pred[:, :, self.config.output_dim:]
        # Compute loss
        loss = self.loss(mean, y, logvar.exp())

        return loss.item()



def train_model(runtime: BaselineConfig, epoch: int, trial: optuna.Trial | None = None) -> None:
    torch.manual_seed(runtime.seed)

    model = Baseline(runtime).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=runtime.learning_rate)

    train_loader, val_loader = make_dataloaders(runtime)
    train_epochs = runtime.epochs if epoch is None else epoch


    before = model.get_loss(x, y)
    print(f"Loss before training: {before:.5f}")

    model.train()
    for step in range(epoch):
        loss = model.train_step(x, y, optimizer)

        if step % 50 == 0:
            print(f"Step {step}: {loss:.5f}")

        if trial is not None:
            trial.report(loss, step)
            if trial.should_prune():
                raise optuna.TrialPruned()

    after = model.get_loss(x, y)
    print(f"Loss after training: {after:.5f}")

    assert torch.isfinite(torch.tensor(after))
    assert after < before, "Model did not improve"







def tune_hyperparameters(config: BaselineConfig) -> BaselineConfig:
    # Placeholder for hyperparameter tuning logic
    # In a real implementation, you would use libraries like Optuna or Ray Tune
    # to perform hyperparameter optimization. Here, we simply return the input config.
    return config

def baseline_train(args):
    torch.manual_seed(args.seed)
    baseline_config = BaselineConfig(**{k: v for k,v in vars(args).items() if k in BaselineConfig.__annotations__})

    runtime = (tune_hyperparameters(baseline_config) if args.tune else baseline_config)

    train_model(runtime)




def main():

    from torch.optim import Adam

    torch.manual_seed(42)

    config = BaselineConfig()

    model = Baseline(config).to(device)

    optimizer = Adam(
        model.parameters(),
        lr=1e-3,
    )

    batch_size = 64
    seq_len = 3
    input_dim = 10

    x = torch.randn(
        batch_size,
        seq_len,
        input_dim,
        device=device,
    )

    # target next observation

    y = torch.randn(
        batch_size,
        1,
        input_dim,
        device=device,
    )
    print(
        "Input:",
        x.shape,
    )

    print(
        "Target:",
        y.shape,
    )

    print(
        "Next state:",
    )

    # ---------------------------------------------------------
    # Initial loss
    # ---------------------------------------------------------

    before = model.get_loss(
        x,
        y,
    )

    print(f"Loss before training: {before:.5f}")

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------

    model.train()

    losses = []

    for step in range(300):
        loss = model.train_step(
            x,
            y,
            optimizer,
        )

        losses.append(loss)

        if step % 50 == 0:
            print(f"Step {step}: {loss:.5f}")

    # ---------------------------------------------------------
    # Final loss
    # ---------------------------------------------------------

    after = model.get_loss(
        x,
        y,
    )

    print(f"Loss after training: {after:.5f}")

    assert torch.isfinite(torch.tensor(after))

    assert after < before, "Model did not improve"

    print("Test passed.")


if __name__ == "__main__":
    main()
 

