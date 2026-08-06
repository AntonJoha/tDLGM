from dataclasses import dataclass
from itertools import chain
import random
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        hidden_dim=1,
        device=None,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
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
                    hidden_dim=config.hidden_dim,
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
        layers=1,
        input_dim=10,
        hidden_dim=1,
        latent_dim=1,
        device=None,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.internal_state = None
        self.latent_dim = latent_dim

        self.first_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            device=device,
        )
        

        self.ffn = nn.Sequential(
            nn.Linear(
                hidden_dim*2,
                hidden_dim,
                device=device,
            ),
            nn.LeakyReLU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
                device=device,
            ),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
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
                hidden_dim,
                device=device,
            ),
            nn.Sigmoid(),
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
                self.hidden_dim,
                device=self.lstm.weight_hh_l0.device,
            ),
            torch.zeros(
                1,
                batch_size,
                self.hidden_dim,
                device=self.lstm.weight_hh_l0.device,
            ),
        )

    def forward(
        self,
        x,
        h,
        xi,
    ):
        
        out, self.internal_state = self.first_lstm(
            x
        )
        #print("internal state:", self.internal_state[0].shape, self.internal_state[1].shape)
        #h, self.internal_state = self.lstm(
            #h,
            #self.internal_state,
        #)
        h = self.ffn(torch.cat([out, h], dim=-1))
        g_xi = 0.1*self.g(xi)
        #print("Magnitude of h:", h.abs().mean().item(), "magnitude of xi:", xi.abs().mean().item(), "magnitude of g(xi):", g_xi.abs().mean().item())
        return h + g_xi


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.layers = config.layers
        self.seq_len = config.seq_len
        self.latent_dim = config.latent_dim

        self.gen_layers = nn.ModuleList(
            [
                GenLayer(
                    layers=config.layers,
                    input_dim=config.input_dim,
                    hidden_dim=config.hidden_dim,
                    latent_dim=config.latent_dim,
                    device=device,
                )
                for _ in range(config.layers)
            ]
        )

        self.initial_transform = nn.Sequential(
            nn.Linear(
                config.latent_dim,
                config.hidden_dim,
                device=device,
            ),
        )

        self.output_layer = nn.Sequential(
            nn.Linear(
                config.hidden_dim,
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
        #self.xi = [torch.zeros_like(x) for x in self.xi]

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
        x,
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
                x,v,
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

        return mean, log_var, z


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
        logvars = []
        zs = []

        for layer in self.rec_layers:
            mean, logvar, z = layer(x)

            means.append(mean)
            logvars.append(logvar)
            zs.append(z)

        return means, logvars, zs


# ── Prior ─────────────────────────────────────────────────────────────────────


class Prior(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.prior_layers = nn.ModuleList(
            [RecLayer(config) for _ in range(config.layers + 1)]
        )

    def sample_xi(self, x):
        prior_mean, prior_logvar = self(x)
        z = []
        for m, lv in zip(
            prior_mean,
            prior_logvar,):
            std = torch.exp(0.5 * lv)
            eps = torch.randn_like(std)
            z.append(m + eps * std)
        return z

    def forward(self, x):

        means = []
        logvars = []

        for layer in self.prior_layers:

            mean = layer.mean(x)
            logvar = layer.log_var(x)

            means.append(mean)
            logvars.append(logvar)

        return means, logvars


# ── tDLGM ─────────────────────────────────────────────────────────────────────


class TDLGM(nn.Module):
    def __init__(
        self,
        config: TDLGMConfig,
    ):
        super().__init__()

        self.config = config

        self.model_t = TimeRecognition(config)

        self.model_g = Generator(config)

        self.model_r = Recognition(config)

        self.model_p = Prior(config)

        self.mse = nn.MSELoss()
        self.loss = nn.GaussianNLLLoss()
        self.kl_multiplier = 0.1

    def get_parameters(self):

        return list(
            chain(
                self.model_t.parameters(),
                self.model_g.parameters(),
                self.model_r.parameters(),
                self.model_p.parameters()
            )
        )

    def gaussian_kl(
    self,
    q_mean,
    q_logvar,
    p_mean,
    p_logvar,):
        q_var = torch.exp(q_logvar)
        p_var = torch.exp(p_logvar)

        kl = 0.5 * (
            p_logvar
            - q_logvar
            + (q_var + (q_mean - p_mean).pow(2))
              / p_var
            - 1
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
        
        return 0.5 * torch.sum(
            torch.exp(logvar) + mean**2 - 1.0 - logvar,
            dim=-1,
        ).mean()



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
        mean_q,
        logvar_q,
        generated_state,
        target_state,
        mean_p,
        logvar_p
    ):

        # Gaussian NLL: 0.5 * (log_var + (y - mean)^2 / var)

        if pred_mean.ndim == 2:
            pred_mean = pred_mean.unsqueeze(1)
        if y.ndim == 2:
            y = y.unsqueeze(1)
        if y.shape != pred_mean.shape:
            pred_mean = pred_mean.expand_as(y)
            pred_log_var = pred_log_var.expand_as(y)
        y_flat = y.reshape_as(pred_mean)
        reconstruction = (
            0.5
            * (pred_log_var + (y_flat - pred_mean).pow(2) / pred_log_var.exp()).mean()
        )

        kl = 0.0
        for mq, lvq, mp, lvp in zip(mean_q,logvar_q,mean_p,logvar_p):
            kl += self.gaussian_kl(mq, lvq, mp, lvp,)


        kl /= len(mean_q)

        consistency = self.state_loss(
            generated_state,
            target_state,
        )
        self.kl_multiplier *= 1.1
        self.kl_multiplier = min(self.kl_multiplier, 1.0)
        #print("rec:", reconstruction.item(), "kl:", kl.item())
        return reconstruction + self.kl_multiplier* kl # + consistency*0.1

    

    def train_step(
        self,
        x,
        y,
        optimizer,
    ):

        optimizer.zero_grad()

        # encode previous state

        t = self.model_t(x)
        
        x_1 = torch.cat(
            [
                x[:, 1:, :],
                y,
            ],
            dim=1,
        )

        t_1 = self.model_t(x_1)

        self.model_g.make_internal_state(x.size(0))

        self.model_g.set_internal_state(t)

        # infer latent noise

        mean, R, z = self.model_r(x_1)
        mean_p, logvar_p = self.model_p(x)
        
        if random.random() < 0.8:
            self.model_g.set_xi(self.model_p.sample_xi(x))
        else:
            self.model_g.set_xi(z)

        pred_mean, pred_log_var, state = self.model_g(x, x.size(0))

        loss = self.compute_loss(
            y,
            pred_mean,
            pred_log_var,
            mean,
            R,
            state,
            t_1,
            mean_p,
            logvar_p
        )

        loss.backward()


        optimizer.step()

        return loss.item()

    def _compute_losses(
            self,
        y,
        pred_mean,
        pred_log_var,
        mean_q,
        logvar_q,
        generated_state,
        target_state,
        mean_p,
        logvar_p
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
        for mq, lvq, mp, lvp in zip(mean_q,logvar_q,mean_p,logvar_p):
            kl += self.gaussian_kl(mq, lvq, mp, lvp,)


        kl /= len(mean_q)

        consistency = self.state_loss(
            generated_state,
            target_state,
        )

        return reconstruction, self.kl_multiplier*kl, consistency
    

    @torch.no_grad()
    def compute_losses(self, x,y, prior=True):

        # encode previous state


        t = self.model_t(x)
        
        x_1 = torch.cat(
            [
                x[:, 1:, :],
                y,
            ],
            dim=1,
        )

        t_1 = self.model_t(x_1)

        self.model_g.make_internal_state(x.size(0))

        self.model_g.set_internal_state(t)

        # infer latent noise

        mean, R, z = self.model_r(x_1)
        mean_p, logvar_p = self.model_p(x)

        if prior:
            self.model_g.set_xi(self.model_p.sample_xi(x))
        else:
            self.model_g.set_xi(z)

        pred_mean, pred_log_var, state = self.model_g(x, x.size(0))

        rec, kl, consistency = self._compute_losses(
            y,
            pred_mean,
            pred_log_var,
            mean,
            R,
            state,
            t_1,
            mean_p,
            logvar_p
        )


        return rec.item(), kl.item(), consistency.item()




    def forward(self, x):

        t = self.model_t(x)
        self.model_g.make_internal_state(x.size(0))

        self.model_g.set_internal_state(t)

        #if x_1 is not None:
            #_, _, z = self.model_r(x)
            #self.model_g.set_xi(z)
        #else:
            #self.model_g.set_xi(None)

        #mean, logvar, z = self.model_r(x)
        xi = self.model_p.sample_xi(x)
        self.model_g.set_xi(xi)

        pred_mean, pred_log_var, _ = self.model_g(x, x.size(0))

        return pred_mean, pred_log_var, 0, 0, 0

    @torch.no_grad()
    def get_loss(
        self,
        x,
        y,
    ):

        self.eval()

        t = self.model_t(x)

        x_1 = torch.cat(
            [
                x[:, 1:, :],
                y,
                ],
            dim=1,
            )
    
        t_1 = self.model_t(x_1)

        self.model_g.make_internal_state(x.size(0))

        self.model_g.set_internal_state(t)

        mean, R, z = self.model_r(x_1)

        # print(z)

        self.model_g.set_xi(z)

        pred_mean, pred_log_var, state = self.model_g(x, x.size(0))

        mean_p, logvar_p = self.model_p(x)

        return self.compute_loss(
            y,
            pred_mean,
            pred_log_var,
            mean,
            R,
            state,
            t_1,
            mean_p,
            logvar_p,
        ).item()


    def nllLoss(self, mean, y, logvar):
        return (
            0.5
            * (
                torch.exp(-logvar)
                * (y - mean).pow(2)
                + logvar
            )
        ).mean()





# ── Self Test ────────────────────────────────────────────────────────────────


def main():


    torch.manual_seed(42)

    config = TDLGMConfig()

    model = TDLGM(config).to(device)

    optimizer = SGD(
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
