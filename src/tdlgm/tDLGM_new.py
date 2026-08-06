import logging
from dataclasses import dataclass
from itertools import chain

import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TDLGMConfig:
    # Architecture
    input_dim: int = 10
    hidden_dim: int = 20
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


# ── tDLGM ─────────────────────────────────────────────────────────────────────


class TDLGM(nn.Module):
    def __init__(
        self,
        config: TDLGMConfig,
    ):
        super().__init__()

        self.config = config

        self.model_t = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.layers,
            batch_first=True,
        )
        self.model_g = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        self.model_r = nn.Sequential(
            nn.Linear(
                config.input_dim * (1 + config.seq_len) + config.hidden_dim,
                config.hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.latent_dim * 2),
        )

        self.model_p = nn.Sequential(
            nn.Linear(config.input_dim * config.seq_len, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.latent_dim * 2),
        )
        self.model_mean = nn.Linear(config.hidden_dim, config.output_dim)
        self.model_logvar = nn.Linear(config.hidden_dim, config.output_dim)

        self.mse = nn.MSELoss()
        self.loss = nn.GaussianNLLLoss()
        self.kl_multiplier = 0.001

    def forward(self, x, y=None):

        t, _ = self.model_t(x)
        t = t[:, -1, :]
        if y is not None:
            x_1 = torch.cat(
                [
                    x[:, 1:, :],
                    y,
                ],
                dim=1,
            )
            xi = self.model_r(
                torch.cat(
                    [
                        x_1.flatten(-2, -1),
                        t,
                    ],
                    dim=-1,
                )
            )
        else:
            xi = self.model_p(x.flatten(-2, -1))
        xi_mean, xi_log_var = torch.chunk(xi, 2, dim=-1)
        z = self.reparameterize(xi_mean, xi_log_var)

        h_t = t + self.model_g(z)
        pred_mean = self.model_mean(h_t)
        pred_logvar = self.model_logvar(h_t)

        return pred_mean, pred_logvar, 0, 0, 0

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def get_parameters(self):

        return list(
            chain(
                self.model_t.parameters(),
                self.model_g.parameters(),
                self.model_r.parameters(),
                self.model_p.parameters(),
            )
        )

    def gaussian_kl(
        self,
        q_mean,
        q_logvar,
        p_mean,
        p_logvar,
    ):
        q_var = torch.exp(q_logvar)
        p_var = torch.exp(p_logvar)

        kl = 0.5 * (
            p_logvar - q_logvar + (q_var + (q_mean - p_mean).pow(2)) / p_var - 1
        )

        return kl.sum(dim=-1).mean()

    def gaussian_kl_old(
        self,
        mean,
        logvar,
    ):
        """
        KL(q(z)||N(0,I))

        q(z)=N(mean, RR^T)

        """

        return (
            0.5
            * torch.sum(
                torch.exp(logvar) + mean**2 - 1.0 - logvar,
                dim=-1,
            ).mean()
        )

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
        self, y, pred_mean, pred_log_var, mean_q, logvar_q, mean_p, logvar_p
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
        for mq, lvq, mp, lvp in zip(mean_q, logvar_q, mean_p, logvar_p):
            kl += self.gaussian_kl(
                mq,
                lvq,
                mp,
                lvp,
            )

        kl /= len(mean_q)

        self.kl_multiplier *= 1.1
        self.kl_multiplier = min(self.kl_multiplier, 1.0)
        return reconstruction + self.kl_multiplier * kl

    def train_step(
        self,
        x,
        y,
        optimizer,
    ):

        optimizer.zero_grad()

        # encode previous state

        t, _ = self.model_t(x)
        t = t[:, -1, :]

        x_1 = torch.cat(
            [
                x,
                y,
            ],
            dim=1,
        )
        xi_r = self.model_r(
            torch.cat(
                [
                    x_1.flatten(-2, -1),
                    t,
                ],
                dim=-1,
            )
        )

        mean, logvar = torch.chunk(xi_r, 2, dim=-1)

        xi_p = self.model_p(x.flatten(-2, -1))
        mean_p, logvar_p = torch.chunk(xi_p, 2, dim=-1)

        xi = xi_r

        xi_mean, xi_log_var = torch.chunk(xi, 2, dim=-1)
        z = self.reparameterize(xi_mean, xi_log_var)

        h_t = t + self.model_g(z)
        pred_mean = self.model_mean(h_t)
        pred_logvar = self.model_logvar(h_t)

        loss = self.compute_loss(
            y, pred_mean, pred_logvar, mean, logvar, mean_p, logvar_p
        )

        loss.backward()

        optimizer.step()

        return loss.item()

    def _compute_losses(
        self, y, pred_mean, pred_log_var, mean_q, logvar_q, mean_p, logvar_p
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
        for mq, lvq, mp, lvp in zip(mean_q, logvar_q, mean_p, logvar_p):
            kl += self.gaussian_kl(
                mq,
                lvq,
                mp,
                lvp,
            )

        kl /= len(mean_q)

        return reconstruction, self.kl_multiplier * kl

    @torch.no_grad()
    def compute_losses(self, x, y, prior=True):

        # encode previous state

        t, _ = self.model_t(x)
        t = t[:, -1, :]

        x_1 = torch.cat(
            [
                x,
                y,
            ],
            dim=1,
        )
        xi_r = self.model_r(
            torch.cat(
                [
                    x_1.flatten(-2, -1),
                    t,
                ],
                dim=-1,
            )
        )
        mean, logvar = torch.chunk(xi_r, 2, dim=-1)

        xi_p = self.model_p(x.flatten(-2, -1))
        mean_p, logvar_p = torch.chunk(xi_p, 2, dim=-1)

        if prior:
            xi = xi_p
        else:
            xi = xi_r

        xi_mean, xi_log_var = torch.chunk(xi, 2, dim=-1)
        z = self.reparameterize(xi_mean, xi_log_var)

        h_t = t + self.model_g(z)
        pred_mean = self.model_mean(h_t)
        pred_logvar = self.model_logvar(h_t)

        rec, kl = self._compute_losses(
            y, pred_mean, pred_logvar, mean, logvar, mean_p, logvar_p
        )

        return rec.item(), kl.item(), 0

    @torch.no_grad()
    def get_loss(self, x, y, prior=True):

        self.eval()

        t, _ = self.model_t(x)
        t = t[:, -1, :]

        x_1 = torch.cat(
            [
                x,
                y,
            ],
            dim=1,
        )
        xi_r = self.model_r(x_1.flatten(-2, -1))
        mean, logvar = torch.chunk(xi_r, 2, dim=-1)

        xi_p = self.model_p(x.flatten(-2, -1))
        mean_p, logvar_p = torch.chunk(xi_p, 2, dim=-1)

        if prior:
            xi = xi_p
        else:
            xi = xi_r

        xi_mean, xi_log_var = torch.chunk(xi, 2, dim=-1)
        z = self.reparameterize(xi_mean, xi_log_var)

        h_t = t + self.model_g(z)
        pred_mean = self.model_mean(h_t)
        pred_logvar = self.model_logvar(h_t)

        return self.compute_loss(
            y,
            pred_mean,
            pred_logvar,
            mean,
            logvar,
            mean_p,
            logvar_p,
        ).item()

    def nllLoss(self, mean, y, logvar):
        return (0.5 * (torch.exp(-logvar) * (y - mean).pow(2) + logvar)).mean()
