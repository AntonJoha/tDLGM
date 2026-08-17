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
    horizon: int = 1

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

        self.input_generator = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        self.input_posterior = nn.Sequential(
            nn.Linear(
                config.input_dim * (config.seq_len + config.horizon),
                config.hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.latent_dim * 2),
        )

        self.input_prior = nn.Sequential(
            nn.Linear(config.input_dim * config.seq_len, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.latent_dim * 2),
        )

        self.input_time_latent = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.layers,
            batch_first=True,
        )

        self.model_layers = self.make_tdlmg_layers(config)

        self.model_mean = nn.Linear(config.hidden_dim, config.output_dim * config.horizon)
        self.model_logvar = nn.Linear(config.hidden_dim, config.output_dim * config.horizon)

        self.mse = nn.MSELoss()
        self.nllLoss = nn.GaussianNLLLoss()
        self.kl_multiplier = 1

    def make_tdlgm_layer(self, config):

        combinator = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        generator = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        time_latent = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.layers,
            batch_first=True,
        )

        posterior = nn.Sequential(
            nn.Linear(
                config.input_dim * (config.seq_len + config.horizon),
                config.hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.latent_dim * 2),
        )

        prior = nn.Sequential(
            nn.Linear(config.input_dim * config.seq_len, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.latent_dim * 2),
        )
        return nn.ModuleDict(
            {
                "combinator": combinator,
                "time_latent": time_latent,
                "generator": generator,
                "posterior": posterior,
                "prior": prior,
            }
        )

    def make_tdlmg_layers(self, config):
        layers = nn.ModuleList()
        for _ in range(config.tdlgm_layers):
            layers.append(self.make_tdlgm_layer(config))
        return nn.Sequential(*layers)


def _to_output_shape(self, x):
    x = x.view(x.size(0), self.config.horizon, self.config.output_dim)
    return x.squeeze(-1) if self.config.output_dim == 1 else x

    def forward(self, x):

        t, _ = self.input_time_latent(x)
        t = t[:, -1, :]

        xi_p = self.input_prior(x.flatten(-2, -1))
        mean_p, logvar_p = torch.chunk(xi_p, 2, dim=-1)
        h = self.input_generator(self.reparameterize(mean_p, logvar_p))

        for layer in self.model_layers:
            t, _ = layer["time_latent"](x)
            t = t[:, -1, :]

            xi_p = layer["prior"](x.flatten(-2, -1))
            mean_p, logvar_p = torch.chunk(xi_p, 2, dim=-1)

            xi = xi_p

            xi_mean, xi_log_var = torch.chunk(xi, 2, dim=-1)
            z = self.reparameterize(xi_mean, xi_log_var)

            h = layer["combinator"](torch.cat([t, h], dim=-1)) + layer["generator"](z)
        pred_mean = self._to_output_shape(self.model_mean(h))
        pred_logvar = self._to_output_shape(self.model_logvar(h))

        return pred_mean, pred_logvar

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
        self,
        y,
        pred_mean,
        pred_logvar,
        mean_q_list,
        logvar_q_list,
        mean_p_list,
        logvar_p_list,
    ):

        rec, kl = self._compute_losses(
            y,
            pred_mean,
            pred_logvar,
            mean_q_list,
            logvar_q_list,
            mean_p_list,
            logvar_p_list,
        )

        return rec + kl

    def train_step(
        self,
        x,
        y,
        optimizer,
    ):

        optimizer.zero_grad()

        # Encodings needed for KL divergence
        mean_q_list = []
        logvar_q_list = []
        mean_p_list = []
        logvar_p_list = []

        t, _ = self.input_time_latent(x)
        t = t[:, -1, :]

        xi_q = self.input_posterior(torch.cat([x, y], dim=-2).flatten(-2, -1))
        mean_q, logvar_q = torch.chunk(xi_q, 2, dim=-1)

        xi_p = self.input_prior(x.flatten(-2, -1))
        mean_p, logvar_p = torch.chunk(xi_p, 2, dim=-1)

        mean_q_list.append(mean_q)
        logvar_q_list.append(logvar_q)
        mean_p_list.append(mean_p)
        logvar_p_list.append(logvar_p)

        h = self.input_generator(
            self.reparameterize(mean_q, logvar_q)
        )  # USE POSTERIOR DURING TRAINING

        for layer in self.model_layers:
            t, _ = layer["time_latent"](x)
            t = t[:, -1, :]

            xi_q = layer["posterior"](torch.cat([x, y], dim=-2).flatten(-2, -1))
            mean_q, logvar_q = torch.chunk(xi_q, 2, dim=-1)
            mean_q_list.append(mean_q)
            logvar_q_list.append(logvar_q)

            xi_p = layer["prior"](x.flatten(-2, -1))
            mean_p, logvar_p = torch.chunk(xi_p, 2, dim=-1)
            mean_p_list.append(mean_p)
            logvar_p_list.append(logvar_p)

            z = self.reparameterize(mean_q, logvar_q)
            h = layer["combinator"](torch.cat([t, h], dim=-1)) + layer["generator"](z)

        pred_mean = self._to_output_shape(self.model_mean(h))
        pred_logvar = self._to_output_shape(self.model_logvar(h))
        loss = self.compute_loss(
            y,
            pred_mean,
            pred_logvar,
            mean_q_list,
            logvar_q_list,
            mean_p_list,
            logvar_p_list,
        )

        loss.backward()

        optimizer.step()

        return loss.item()

    def _compute_losses(
        self,
        y,
        pred_mean,
        pred_logvar,
        mean_q_list,
        logvar_q_list,
        mean_p_list,
        logvar_p_list,
    ):
        target = y.squeeze(-1)
        if pred_mean.shape != target.shape:
            raise ValueError(
                "prediction and target shapes must match "
                "(output_dim should equal horizon): "
                f"{pred_mean.shape} != {target.shape}"
            )

        rec = self.nllLoss(pred_mean, target, pred_logvar.exp())

        kl = 0.0
        for mean_q, logvar_q, mean_p, logvar_p in zip(
            mean_q_list, logvar_q_list, mean_p_list, logvar_p_list
        ):
            kl += self.gaussian_kl(mean_q, logvar_q, mean_p, logvar_p)
        return rec, kl * self.kl_multiplier

    @torch.no_grad()
    def compute_losses(self, x, y, prior=True):

        # Encodings needed for KL divergence
        mean_q_list = []
        logvar_q_list = []
        mean_p_list = []
        logvar_p_list = []

        t, _ = self.input_time_latent(x)
        t = t[:, -1, :]

        xi_q = self.input_posterior(torch.cat([x, y], dim=-2).flatten(-2, -1))
        mean_q, logvar_q = torch.chunk(xi_q, 2, dim=-1)

        xi_p = self.input_prior(x.flatten(-2, -1))
        mean_p, logvar_p = torch.chunk(xi_p, 2, dim=-1)

        mean_q_list.append(mean_q)
        logvar_q_list.append(logvar_q)
        mean_p_list.append(mean_p)
        logvar_p_list.append(logvar_p)

        if prior:
            h = self.input_generator(self.reparameterize(mean_p, logvar_p))
        else:
            h = self.input_generator(self.reparameterize(mean_q, logvar_q))

        for layer in self.model_layers:
            t, _ = layer["time_latent"](x)
            t = t[:, -1, :]

            xi_q = layer["posterior"](torch.cat([x, y], dim=-2).flatten(-2, -1))
            mean_q, logvar_q = torch.chunk(xi_q, 2, dim=-1)
            mean_q_list.append(mean_q)
            logvar_q_list.append(logvar_q)

            xi_p = layer["prior"](x.flatten(-2, -1))
            mean_p, logvar_p = torch.chunk(xi_p, 2, dim=-1)
            mean_p_list.append(mean_p)
            logvar_p_list.append(logvar_p)

            if prior:
                z = self.reparameterize(mean_p, logvar_p)
            else:
                z = self.reparameterize(mean_q, logvar_q)

            h = layer["combinator"](torch.cat([t, h], dim=-1)) + layer["generator"](z)

        pred_mean = self._to_output_shape(self.model_mean(h))
        pred_logvar = self._to_output_shape(self.model_logvar(h))
        rec, kl = self._compute_losses(
            y,
            pred_mean,
            pred_logvar,
            mean_q_list,
            logvar_q_list,
            mean_p_list,
            logvar_p_list,
        )

        return rec.item(), kl.item(), 0

