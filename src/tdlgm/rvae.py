from dataclasses import dataclass
from itertools import chain

import torch
import torch.nn.functional as F
from torch import nn



device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Config
# ============================================================

@dataclass(slots=True)
class TDLGMConfig:
    input_dim: int = 10

    hidden_dim: int = 128
    latent_dim: int = 32

    beta: float = 1.0

    learning_rate: float = 1e-3


# ============================================================
# Utilities
# ============================================================

def reparameterize(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mean + eps * std


def gaussian_kl(
    q_mean,
    q_logvar,
    p_mean,
    p_logvar,
):
    q_var = torch.exp(q_logvar)
    p_var = torch.exp(p_logvar)

    kl = 0.5 * (
        p_logvar
        - q_logvar
        + (q_var + (q_mean - p_mean).pow(2))
        / p_var
        - 1.0
    )

    return kl.sum(dim=-1).mean()


# ============================================================
# Deterministic Recurrent Core
# ============================================================

class RecurrentCore(nn.Module):
    """
    h_t = GRU(
        h_{t-1},
        [z_{t-1}, s_{t-1}, a_{t-1}]
    )
    """

    def __init__(self, cfg):
        super().__init__()

        self.gru = nn.GRUCell(
            cfg.latent_dim + cfg.input_dim,
            cfg.hidden_dim,
        )

    def forward(
        self,
        h,
        z_prev,
        s_prev,
    ):
        x = torch.cat(
            [
                z_prev,
                s_prev,
            ],
            dim=-1,
        )

        return self.gru(x, h)


# ============================================================
# Prior p(z_t | h_t)
# ============================================================

class Prior(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                cfg.hidden_dim,
                256,
            ),
            nn.ReLU(),
            nn.Linear(
                256,
                2 * cfg.latent_dim,
            ),
        )

    def forward(self, h):

        mean, logvar = self.net(h).chunk(
            2,
            dim=-1,
        )

        logvar = logvar.clamp(-10, 10)

        return mean, logvar


# ============================================================
# Posterior q(z_t | h_t, s_t)
# ============================================================

class Posterior(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                cfg.hidden_dim
                + cfg.input_dim,
                256,
            ),
            nn.ReLU(),
            nn.Linear(
                256,
                2 * cfg.latent_dim,
            ),
        )

    def forward(
        self,
        h,
        s,
    ):
        x = torch.cat(
            [
                h,
                s,
            ],
            dim=-1,
        )

        mean, logvar = self.net(x).chunk(
            2,
            dim=-1,
        )

        logvar = logvar.clamp(-10, 10)

        return mean, logvar


# ============================================================
# Decoder p(s_t | h_t, z_t)
# ============================================================

class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.base = nn.Sequential(
            nn.Linear(
                cfg.hidden_dim
                + cfg.latent_dim,
                256,
            ),
            nn.ReLU(),
        )

        self.mean_head = nn.Linear(
            256,
            cfg.input_dim,
        )

        self.var_head = nn.Linear(
            256,
            cfg.input_dim,
        )

    def forward(
        self,
        h,
        z,
    ):
        e = self.base(
            torch.cat(
                [
                    h,
                    z,
                ],
                dim=-1,
            )
        )

        mean = self.mean_head(e)

        var = (
            F.softplus(
                self.var_head(e)
            )
            + 1e-4
        )

        logvar = torch.log(var)

        return mean, logvar


# ============================================================
# Recurrent DLGM / RSSM
# ============================================================

class TDLGM(nn.Module):
    def __init__(
        self,
        config: TDLGMConfig,
    ):
        super().__init__()

        self.config = config

        self.core = RecurrentCore(config)

        self.prior = Prior(config)

        self.posterior = Posterior(config)

        self.decoder = Decoder(config)
        print("rvae initialized with config:")

    def get_parameters(self):
        return chain(
            self.core.parameters(),
            self.prior.parameters(),
            self.posterior.parameters(),
            self.decoder.parameters(),
        )

    # --------------------------------------------------------
    # Training ELBO
    # --------------------------------------------------------

    def forward_train(
        self,
        x, y, optimizer
    ):
        """
        states:
            (B,T,input_dim)

        """

        h_t = self.core(
            torch.zeros(
                x.size(0),
                self.config.hidden_dim,
                device=x.device,
            ),
            torch.zeros(
                x.size(0),
                self.config.latent_dim,
                device=x.device,
            ),
            torch.zeros(
                x.size(0),
                self.config.input_dim,
                device=x.device,



        return {
            "loss": loss,
            "reconstruction": total_rec,
            "kl": total_kl,
        }

    # --------------------------------------------------------
    # Convenience training step
    # --------------------------------------------------------

    def train_step(
        self,
        states,
        optimizer,
    ):
        optimizer.zero_grad()

        out = self.forward_train(
            states,
        )

        out["loss"].backward()

        optimizer.step()

        return {
            k: v.item()
            for k, v in out.items()
        }

    # --------------------------------------------------------
    # Encode history
    # --------------------------------------------------------

    @torch.no_grad()
    def encode_history(
        self,
        states,
    ):
        """
        Returns final belief state h,z.

        states:
            (B,T,input_dim)

        """

        B, T, _ = states.shape

        h = torch.zeros(
            B,
            self.config.hidden_dim,
            device=states.device,
        )

        z_prev = torch.zeros(
            B,
            self.config.latent_dim,
            device=states.device,
        )

        for t in range(T):

            if t > 0:
                h = self.core(
                    h,
                    z_prev,
                    states[:, t - 1])

            post_mean, post_logvar = self.posterior(
                h,
                states[:, t],
            )

            z_prev = reparameterize(
                post_mean,
                post_logvar,
            )

        return h, z_prev

    # --------------------------------------------------------
    # Imaginary rollout
    # --------------------------------------------------------

    # --------------------------------------------------------
    # One-step prediction
    # --------------------------------------------------------

    def nllLoss(self, mean, target, logvar):
        return 0.5 * (logvar + (target - mean).pow(2) / logvar.exp()).sum(-1).mean()

    

    @torch.no_grad()
    def forward(
        self,
        states,
    ):
        h, z = self.encode_history(
            states,
        )

        last_state = states[:, -1]

        h = self.core(
            h,
            z,
            last_state,
        )

        prior_mean, prior_logvar = self.prior(h)

        z = reparameterize(
            prior_mean,
            prior_logvar,
        )
        mean, logvar = self.decoder(
            h,
            z,
        )
        return mean, logvar , 0,0,0
