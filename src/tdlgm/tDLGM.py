from dataclasses import dataclass
from itertools import chain

import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(slots=True)
class tDLGMConfig:
    # Architecture
    input_dim: int = 10
    hidden_size: int = 20
    latent_dim: int = 5
    output_dim: int = 10
    layers: int = 2
    seq_len: int = 3

    # Training
    learning_rate: float = 1e-3
    batch_size: int = 64

    # Misc
    seed: int = 42
    device: str | None = None


# ── Time Recognition ──────────────────────────────────────────────────────────


class TimeLayer(nn.Module):
    """
    Encodes an observed sequence into a latent temporal state.

    The returned hidden/cell states are treated as the latent state
    representation of the sequence.
    """

    def __init__(
        self,
        input_dim=10,
        hidden_size=1,
        device=None,
    ):
        super().__init__()

        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            device=device,
        )

    def forward(self, x):
        _, (h, c) = self.lstm(x)

        # Remove sequence dimension:
        # (1,batch,hidden) -> (batch,hidden)
        return (
            h.squeeze(0),
            c.squeeze(0),
        )


class TimeRecognition(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.layers = config.layers

        self.time_layers = nn.ModuleList(
            [
                TimeLayer(
                    input_dim=config.input_dim,
                    hidden_size=config.hidden_size,
                    device=device,
                )
                for _ in range(config.layers)
            ]
        )

    def forward(self, x):

        return [layer(x) for layer in self.time_layers]


# ── Generator ─────────────────────────────────────────────────────────────────


class GenLayer(nn.Module):
    def __init__(
        self,
        hidden_size=1,
        latent_dim=1,
        device=None,
    ):
        super().__init__()

        self.hidden_size = hidden_size

        self.internal_state = None

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            device=device,
        )

        self.g = nn.Sequential(
            nn.Linear(
                latent_dim,
                latent_dim,
                device=device,
            ),
            nn.LeakyReLU(),
            nn.Linear(
                latent_dim,
                hidden_size,
                device=device,
            ),
            nn.Tanh(),
        )

    def get_internal_state(self):

        if self.internal_state is None:
            return None

        h, c = self.internal_state

        return (
            h.squeeze(0),
            c.squeeze(0),
        )

    def set_internal_state(
        self,
        state,
    ):

        if state is None:
            self.internal_state = None
            return

        h, c = state

        # Convert latent state representation back to
        # LSTM format:
        #
        # (batch,hidden)
        # ->
        # (1,batch,hidden)

        self.internal_state = (
            h.unsqueeze(0),
            c.unsqueeze(0),
        )

    def make_internal_state(
        self,
        batch_size,
    ):

        self.internal_state = (
            torch.zeros(
                1,
                batch_size,
                self.hidden_size,
                device=self.lstm.weight_hh_l0.device,
            ),
            torch.zeros(
                1,
                batch_size,
                self.hidden_size,
                device=self.lstm.weight_hh_l0.device,
            ),
        )

    def forward(
        self,
        h,
        xi,
    ):

        h, self.internal_state = self.lstm(
            h,
            self.internal_state,
        )

        return h + self.g(xi)


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.layers = config.layers
        self.seq_len = config.seq_len
        self.latent_dim = config.latent_dim

        self.gen_layers = nn.ModuleList(
            [
                GenLayer(
                    config.hidden_size,
                    config.latent_dim,
                    device,
                )
                for _ in range(config.layers)
            ]
        )

        self.initial_transform = nn.Sequential(
            nn.Linear(
                config.latent_dim,
                config.hidden_size,
                device=device,
            ),
            nn.Tanh(),
        )

        self.output_layer = nn.Sequential(
            nn.Linear(
                config.hidden_size,
                2 * config.output_dim,
                device=device,
            )
        )

        self.xi = None

    def make_xi(
        self,
        batch_size,
        device,
    ):

        self.xi = [
            torch.randn(
                batch_size,
                self.seq_len,
                self.latent_dim,
                device=device,
            )
            for _ in range(self.layers + 1)
        ]

    def set_xi(
        self,
        xi,
    ):

        self.xi = xi

    def make_internal_state(
        self,
        batch_size,
    ):

        for layer in self.gen_layers:
            layer.make_internal_state(batch_size)

    def set_internal_state(
        self,
        state,
    ):

        for layer, s in zip(
            self.gen_layers,
            state,
            strict=False,
        ):
            layer.set_internal_state(s)

    def get_internal_state(self):

        return [layer.get_internal_state() for layer in self.gen_layers]

    def forward(
        self,
        batch_size,
    ):

        if self.xi is None:
            self.make_xi(
                batch_size,
                next(self.parameters()).device,
            )

        v = self.initial_transform(self.xi[0])

        for i, layer in enumerate(self.gen_layers):
            v = layer(
                v,
                self.xi[i + 1],
            )

        output = self.output_layer(v[:, -1, :])

        # Split into predicted mean and log-variance
        pred_mean, pred_log_var = output.chunk(2, dim=-1)
        pred_log_var = pred_log_var.clamp(min=-100, max=100)

        return (
            pred_mean,
            pred_log_var,
            self.get_internal_state(),
        )


# ── Recognition ───────────────────────────────────────────────────────────────


class RecLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.latent_dim = config.latent_dim

        self.mean = nn.Sequential(
            nn.Linear(
                config.input_dim,
                config.latent_dim,
                device=config.device,
            ),
            nn.Tanh(),
        )

        self.log_var = nn.Sequential(
            nn.Linear(
                config.input_dim,
                config.latent_dim,
                device=device,
            )
        )

    def forward(self, x):
        mean = self.mean(x)
        log_var = self.log_var(x).clamp(min=-100, max=100)

        std = torch.exp(0.5 * log_var)

        eps = torch.randn_like(std)

        z = mean + eps * std

        return mean, torch.diag_embed(std), z


class Recognition(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.rec_layers = nn.ModuleList(
            [RecLayer(config) for _ in range(config.layers + 1)]
        )

    def forward(
        self,
        x,
    ):

        means = []
        Rs = []
        zs = []

        for layer in self.rec_layers:
            mean, R, z = layer(x)

            means.append(mean)
            Rs.append(R)
            zs.append(z)

        return means, Rs, zs


# ── tDLGM ─────────────────────────────────────────────────────────────────────


class tDLGM(nn.Module):
    def __init__(
        self,
        config: tDLGMConfig,
    ):
        super().__init__()

        self.config = config

        self.model_t = TimeRecognition(config)

        self.model_g = Generator(config)

        self.model_r = Recognition(config)

        self.mse = nn.MSELoss()

    def get_parameters(self):

        return list(
            chain(
                self.model_t.parameters(),
                self.model_g.parameters(),
                self.model_r.parameters(),
            )
        )

    def gaussian_kl(
        self,
        mean,
        R,
    ):
        """
        KL(q(z)||N(0,I))

        q(z)=N(mean, RR^T)

        """

        covariance = R @ R.transpose(
            -1,
            -2,
        )

        trace = torch.diagonal(
            covariance,
            dim1=-2,
            dim2=-1,
        ).sum(-1)

        logdet = 2 * torch.log(
            torch.diagonal(
                R,
                dim1=-2,
                dim2=-1,
            )
        ).sum(-1)

        latent_dim = mean.size(-1)

        kl = 0.5 * (mean.pow(2).sum(-1) + trace - logdet - latent_dim)

        return kl.mean()

    def state_loss(
        self,
        generated_state,
        target_state,
    ):

        loss = 0.0

        for g, t in zip(
            generated_state,
            target_state,
            strict=False,
        ):
            gh, gc = g
            th, tc = t

            loss += self.mse(
                gh,
                th,
            )

            loss += self.mse(
                gc,
                tc,
            )

        return loss

    def compute_loss(
        self,
        y,
        pred_mean,
        pred_log_var,
        mean,
        R,
        generated_state,
        target_state,
    ):

        # Gaussian NLL: 0.5 * (log_var + (y - mean)^2 / var)

        # TODO THIS ONLY SUPPORTS ONE STEP PREDICTION, NEED TO FIX FOR MULTI-STEP
        if y.ndim >= 3 and y.size(1) != 1:
            raise ValueError(f"tDLGM currently supports horizon=1; got {y.size(1)}")

        if pred_mean.ndim == 2:
            pred_mean = pred_mean.unsqueeze(1)
            y = y[:, 0, :]
        y_flat = y.reshape_as(pred_mean)
        reconstruction = (
            0.5
            * (pred_log_var + (y_flat - pred_mean).pow(2) / pred_log_var.exp()).mean()
        )

        kl = 0.0

        for m, r in zip(
            mean,
            R,
            strict=False,
        ):
            kl += self.gaussian_kl(
                m,
                r,
            )

        kl /= len(mean)

        consistency = self.state_loss(
            generated_state,
            target_state,
        )
        return reconstruction + kl + 0.01 * consistency

    def train_step(
        self,
        x,
        x_1,
        y,
        optimizer,
    ):

        optimizer.zero_grad()

        # encode previous state

        t = self.model_t(x)

        t_1 = self.model_t(x_1)

        self.model_g.make_internal_state(x.size(0))

        self.model_g.set_internal_state(t)

        # infer latent noise

        mean, R, z = self.model_r(x_1)

        self.model_g.set_xi(z)

        pred_mean, pred_log_var, state = self.model_g(x.size(0))

        loss = self.compute_loss(
            y,
            pred_mean,
            pred_log_var,
            mean,
            R,
            state,
            t_1,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.parameters(),
            5.0,
        )

        optimizer.step()

        return loss.item()

    @torch.no_grad()
    def get_loss(
        self,
        x,
        x_1,
        y,
    ):

        self.eval()

        t = self.model_t(x)

        t_1 = self.model_t(x_1)

        self.model_g.make_internal_state(x.size(0))

        self.model_g.set_internal_state(t)

        mean, R, z = self.model_r(x_1)

        # print(z)

        self.model_g.set_xi(z)

        pred_mean, pred_log_var, state = self.model_g(x.size(0))

        return self.compute_loss(
            y,
            pred_mean,
            pred_log_var,
            mean,
            R,
            state,
            t_1,
        ).item()


# ── Self Test ────────────────────────────────────────────────────────────────


def main():

    from torch.optim import Adam

    torch.manual_seed(42)

    config = tDLGMConfig()

    model = tDLGM(config).to(device)

    optimizer = Adam(
        model.get_parameters(),
        lr=1e-3,
    )

    # ---------------------------------------------------------
    # Create synthetic sequence prediction problem
    #
    # x_t  -> x_{t+1}
    #
    # ---------------------------------------------------------

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

    # construct x_1:
    #
    # [x_1,x_2,x_3,y]
    #
    # take future sequence

    x_1 = torch.cat(
        [
            x,
            y,
        ],
        dim=1,
    )[:, 1:, :]

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
        x_1.shape,
    )

    # ---------------------------------------------------------
    # Initial loss
    # ---------------------------------------------------------

    before = model.get_loss(
        x,
        x_1,
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
            x_1,
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
        x_1,
        y,
    )

    print(f"Loss after training: {after:.5f}")

    assert torch.isfinite(torch.tensor(after))

    assert after < before, "Model did not improve"

    print("Test passed.")


if __name__ == "__main__":
    main()
