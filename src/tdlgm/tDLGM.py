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
    kl_anneal_steps: int = 1_000
    free_bits: float = 0.0
    posterior_sampling_start: float = 1.0
    posterior_sampling_end: float = 0.1
    sampling_anneal_steps: int = 1_000

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
        self.latent_dim = config.latent_dim
        self.hidden_size = config.hidden_size
        self.cells = nn.ModuleList(
            [
                nn.LSTMCell(
                    config.latent_dim if i == 0 else config.hidden_size,
                    config.hidden_size,
                    device=device,
                )
                for _ in range(config.layers)
            ]
        )
        self.prior = nn.Linear(
            config.hidden_size, 2 * config.latent_dim, device=device
        )
        self.output_layer = nn.Sequential(
            nn.Linear(
                config.hidden_size,
                2 * config.output_dim,
                device=device,
            )
        )

    def initial_state(self, batch_size):
        zeros = torch.zeros(
            batch_size, self.hidden_size, device=next(self.parameters()).device
        )
        return [(zeros, zeros) for _ in self.cells]

    def prior_parameters(self, state):
        mean, log_var = self.prior(state[-1][0]).chunk(2, dim=-1)
        return mean, log_var.clamp(min=-20, max=20)

    def transition(self, z, state):
        next_state = []
        value = z
        for cell, (h, c) in zip(self.cells, state, strict=True):
            h, c = cell(value, (h, c))
            next_state.append((h, c))
            value = h
        return next_state

    def forward(self, z, state):
        means, log_vars = [], []
        for z_t in z.unbind(dim=1):
            state = self.transition(z_t, state)
            mean, log_var = self.output_layer(state[-1][0]).chunk(2, dim=-1)
            means.append(mean)
            log_vars.append(log_var.clamp(min=-20, max=20))
        return torch.stack(means, dim=1), torch.stack(log_vars, dim=1), state

    def sample_prior(self, state, seq_len):
        samples, means, log_vars = [], [], []
        for _ in range(seq_len):
            mean, log_var = self.prior_parameters(state)
            z = mean + torch.randn_like(mean) * torch.exp(0.5 * log_var)
            samples.append(z)
            means.append(mean)
            log_vars.append(log_var)
            state = self.transition(z, state)
        return (
            torch.stack(samples, dim=1),
            torch.stack(means, dim=1),
            torch.stack(log_vars, dim=1),
        )

    def prior_for_latents(self, state, z):
        means, log_vars = [], []
        for z_t in z.unbind(dim=1):
            mean, log_var = self.prior_parameters(state)
            means.append(mean)
            log_vars.append(log_var)
            state = self.transition(z_t, state)
        return torch.stack(means, dim=1), torch.stack(log_vars, dim=1)


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
        self.encoder = nn.LSTM(
            config.input_dim,
            config.hidden_size,
            num_layers=config.layers,
            batch_first=True,
            device=device,
        )
        self.posterior = nn.Linear(
            config.hidden_size, 2 * config.latent_dim, device=device
        )

    def forward(self, x):
        encoded, _ = self.encoder(x)
        mean, log_var = self.posterior(encoded).chunk(2, dim=-1)
        log_var = log_var.clamp(min=-20, max=20)
        z = mean + torch.randn_like(mean) * torch.exp(0.5 * log_var)
        return mean, log_var, z


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
        self.register_buffer("training_steps", torch.zeros((), dtype=torch.long))

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
        log_var,
        prior_mean,
        prior_log_var,
    ):
        """
        KL(q(z|x)||p(z|h)) for diagonal Gaussian posterior and prior.

        """

        posterior_var = log_var.exp()
        prior_var = prior_log_var.exp()
        kl = 0.5 * (
            prior_log_var
            - log_var
            + (posterior_var + (mean - prior_mean).pow(2)) / prior_var
            - 1
        )
        return kl.clamp_min(self.config.free_bits).sum(dim=-1).mean()

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
        target,
        pred_mean,
        pred_log_var,
        mean,
        log_var,
        prior_mean,
        prior_log_var,
    ):

        target = target.reshape_as(pred_mean)
        reconstruction = (
            0.5
            * (
                pred_log_var
                + (target - pred_mean).pow(2) / pred_log_var.exp()
            ).mean()
        )
        kl = self.gaussian_kl(
            mean, log_var, prior_mean, prior_log_var
        )
        beta = min(
            1.0, self.training_steps.item() / max(1, self.config.kl_anneal_steps)
        )
        return reconstruction + beta * kl

    def train_step(
        self,
        x,
        x_1,
        y,
        optimizer,
    ):

        optimizer.zero_grad()

        state = self.model_t(x)
        mean, log_var, posterior_z = self.model_r(x_1)
        prior_z, _, _ = self.model_g.sample_prior(state, x_1.size(1))
        progress = min(
            1.0,
            self.training_steps.item() / max(1, self.config.sampling_anneal_steps),
        )
        posterior_probability = (
            self.config.posterior_sampling_start
            + progress
            * (
                self.config.posterior_sampling_end
                - self.config.posterior_sampling_start
            )
        )
        use_posterior = torch.rand(
            x.size(0), 1, 1, device=x.device
        ) < posterior_probability
        z = torch.where(use_posterior, posterior_z, prior_z)
        prior_mean, prior_log_var = self.model_g.prior_for_latents(state, z)
        pred_mean, pred_log_var, _ = self.model_g(z, state)

        loss = self.compute_loss(
            x_1,
            pred_mean,
            pred_log_var,
            mean,
            log_var,
            prior_mean,
            prior_log_var,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.parameters(),
            5.0,
        )

        optimizer.step()
        self.training_steps += 1

        return loss.item()

    @torch.no_grad()
    def get_loss(
        self,
        x,
        x_1,
        y,
    ):

        self.eval()

        state = self.model_t(x)
        z, prior_mean, prior_log_var = self.model_g.sample_prior(state, x_1.size(1))
        pred_mean, pred_log_var, _ = self.model_g(z, state)

        return self.compute_loss(
            x_1,
            pred_mean,
            pred_log_var,
            prior_mean,
            prior_log_var,
            prior_mean,
            prior_log_var,
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
