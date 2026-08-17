# ruff: noqa: N999

from __future__ import annotations

import torch
from torch import nn

from experiments.util import SeriesConfig

TDLGMConfig = SeriesConfig


def _resolve_num_heads(hidden_dim: int) -> int:
    for candidate in (8, 4, 2):
        if hidden_dim % candidate == 0:
            return candidate
    return 1


class SequenceAttentionEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int, seq_len: int):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=_resolve_num_heads(hidden_dim),
                dim_feedforward=hidden_dim * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=layers,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        if x.size(1) > self.position_embedding.size(1):
            raise ValueError(
                "sequence length exceeds configured maximum: "
                f"{x.size(1)} > {self.position_embedding.size(1)}"
            )
        h = self.input_proj(x) + self.position_embedding[:, : x.size(1), :]
        h = self.encoder(h)
        return self.norm(h)


def _make_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class TDLGM(nn.Module):
    def __init__(self, config: TDLGMConfig):
        super().__init__()
        self.config = config

        self.input_time_latent = SequenceAttentionEncoder(
            config.input_dim,
            config.hidden_dim,
            config.layers,
            config.seq_len,
        )
        self.input_generator = _make_mlp(
            config.latent_dim, config.hidden_dim, config.hidden_dim
        )
        self.input_posterior = _make_mlp(
            config.input_dim * (config.seq_len + config.horizon),
            config.hidden_dim,
            config.latent_dim * 2,
        )
        self.input_prior = _make_mlp(
            config.input_dim * config.seq_len,
            config.hidden_dim,
            config.latent_dim * 2,
        )

        self.model_layers = self.make_tdlgm_layers(config)
        self.model_mean = nn.Linear(
            config.hidden_dim, config.output_dim * config.horizon
        )
        self.model_logvar = nn.Linear(
            config.hidden_dim, config.output_dim * config.horizon
        )

        self.mse = nn.MSELoss()
        self.nllLoss = nn.GaussianNLLLoss()
        self.kl_multiplier = 1.0

    def make_tdlgm_layer(self, config: TDLGMConfig) -> nn.ModuleDict:
        return nn.ModuleDict(
            {
                "combinator": nn.Sequential(
                    nn.Linear(config.hidden_dim * 2, config.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                ),
                "time_latent": SequenceAttentionEncoder(
                    config.input_dim,
                    config.hidden_dim,
                    config.layers,
                    config.seq_len,
                ),
                "generator": _make_mlp(
                    config.latent_dim,
                    config.hidden_dim,
                    config.hidden_dim,
                ),
                "posterior": _make_mlp(
                    config.input_dim * (config.seq_len + config.horizon),
                    config.hidden_dim,
                    config.latent_dim * 2,
                ),
                "prior": _make_mlp(
                    config.input_dim * config.seq_len,
                    config.hidden_dim,
                    config.latent_dim * 2,
                ),
            }
        )

    def make_tdlgm_layers(self, config: TDLGMConfig) -> nn.ModuleList:
        layers = nn.ModuleList()
        for _ in range(config.tdlgm_layers):
            layers.append(self.make_tdlgm_layer(config))
        return layers

    def _to_output_shape(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), self.config.horizon, self.config.output_dim)
        return x.squeeze(-1) if self.config.output_dim == 1 else x

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mean + torch.randn_like(std) * std

    def _sequence_summary(self, encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return encoder(x).mean(dim=1)

    def _latent_pass(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        prior: bool = True,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        if y is not None and y.ndim == 2:
            y = y.unsqueeze(-1)
        x_flat = x.flatten(start_dim=1)
        y_flat = None if y is None else torch.cat([x, y], dim=-2).flatten(start_dim=1)

        mean_q_list: list[torch.Tensor] = []
        logvar_q_list: list[torch.Tensor] = []
        mean_p_list: list[torch.Tensor] = []
        logvar_p_list: list[torch.Tensor] = []

        if y_flat is not None:
            xi_q = self.input_posterior(y_flat)
            mean_q, logvar_q = torch.chunk(xi_q, 2, dim=-1)
            mean_q_list.append(mean_q)
            logvar_q_list.append(logvar_q)
        else:
            mean_q = logvar_q = None

        xi_p = self.input_prior(x_flat)
        mean_p, logvar_p = torch.chunk(xi_p, 2, dim=-1)
        mean_p_list.append(mean_p)
        logvar_p_list.append(logvar_p)

        h = self.input_generator(
            self.reparameterize(mean_p, logvar_p)
            if prior or mean_q is None
            else self.reparameterize(mean_q, logvar_q)
        )

        for layer in self.model_layers:
            t = self._sequence_summary(layer["time_latent"], x)

            if y_flat is not None:
                xi_q = layer["posterior"](y_flat)
                mean_q, logvar_q = torch.chunk(xi_q, 2, dim=-1)
                mean_q_list.append(mean_q)
                logvar_q_list.append(logvar_q)

            xi_p = layer["prior"](x_flat)
            mean_p, logvar_p = torch.chunk(xi_p, 2, dim=-1)
            mean_p_list.append(mean_p)
            logvar_p_list.append(logvar_p)

            z = (
                self.reparameterize(mean_p, logvar_p)
                if prior or mean_q is None
                else self.reparameterize(mean_q, logvar_q)
            )
            h = layer["combinator"](torch.cat([t, h], dim=-1)) + layer["generator"](z)

        pred_mean = self._to_output_shape(self.model_mean(h))
        pred_logvar = self._to_output_shape(self.model_logvar(h))
        return (
            pred_mean,
            pred_logvar,
            mean_q_list,
            logvar_q_list,
            mean_p_list,
            logvar_p_list,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pred_mean, pred_logvar, *_ = self._latent_pass(x, prior=True)
        return pred_mean, pred_logvar

    def get_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.parameters())

    def gaussian_kl(
        self,
        q_mean: torch.Tensor,
        q_logvar: torch.Tensor,
        p_mean: torch.Tensor,
        p_logvar: torch.Tensor,
    ) -> torch.Tensor:
        q_var = torch.exp(q_logvar)
        p_var = torch.exp(p_logvar)
        kl = 0.5 * (
            p_logvar - q_logvar + (q_var + (q_mean - p_mean).pow(2)) / p_var - 1
        )
        return kl.sum(dim=-1).mean()

    def _compute_losses(
        self,
        y: torch.Tensor,
        pred_mean: torch.Tensor,
        pred_logvar: torch.Tensor,
        mean_q_list: list[torch.Tensor],
        logvar_q_list: list[torch.Tensor],
        mean_p_list: list[torch.Tensor],
        logvar_p_list: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

    def compute_loss(
        self,
        y: torch.Tensor,
        pred_mean: torch.Tensor,
        pred_logvar: torch.Tensor,
        mean_q_list: list[torch.Tensor],
        logvar_q_list: list[torch.Tensor],
        mean_p_list: list[torch.Tensor],
        logvar_p_list: list[torch.Tensor],
    ) -> torch.Tensor:
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
        x: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        self.train()
        optimizer.zero_grad()
        (
            pred_mean,
            pred_logvar,
            mean_q_list,
            logvar_q_list,
            mean_p_list,
            logvar_p_list,
        ) = self._latent_pass(x, y, prior=False)
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
        return float(loss)

    @torch.no_grad()
    def get_loss(self, x: torch.Tensor, y: torch.Tensor) -> float:
        (
            pred_mean,
            pred_logvar,
            mean_q_list,
            logvar_q_list,
            mean_p_list,
            logvar_p_list,
        ) = self._latent_pass(x, y, prior=True)
        loss = self.compute_loss(
            y,
            pred_mean,
            pred_logvar,
            mean_q_list,
            logvar_q_list,
            mean_p_list,
            logvar_p_list,
        )
        return float(loss)

    @torch.no_grad()
    def compute_losses(self, x: torch.Tensor, y: torch.Tensor, prior: bool = True):
        (
            pred_mean,
            pred_logvar,
            mean_q_list,
            logvar_q_list,
            mean_p_list,
            logvar_p_list,
        ) = self._latent_pass(x, y, prior=prior)
        rec, kl = self._compute_losses(
            y,
            pred_mean,
            pred_logvar,
            mean_q_list,
            logvar_q_list,
            mean_p_list,
            logvar_p_list,
        )
        return float(rec), float(kl), 0.0


if __name__ == "__main__":
    from .main import main

    main()
