from itertools import chain

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Time Recognition ──────────────────────────────────────────────────────────


class TimeLayer(nn.Module):
    """
    Encodes an observed sequence into a latent temporal state.

    The returned hidden/cell states are treated as the latent state
    representation of the sequence.
    """

    def __init__(
        self,
        input_dim=1,
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
    def __init__(
        self,
        input_dim=1,
        hidden_size=1,
        seq_len=1,
        layers=1,
        device=None,
    ):
        super().__init__()

        self.layers = layers

        self.time_layers = nn.ModuleList(
            [
                TimeLayer(
                    input_dim=input_dim,
                    hidden_size=hidden_size,
                    device=device,
                )
                for _ in range(layers)
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
    def __init__(
        self,
        hidden_size=1,
        latent_dim=1,
        output_dim=1,
        layers=1,
        seq_len=1,
        device=None,
    ):
        super().__init__()

        self.layers = layers
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        self.gen_layers = nn.ModuleList(
            [
                GenLayer(
                    hidden_size,
                    latent_dim,
                    device,
                )
                for _ in range(layers)
            ]
        )

        self.initial_transform = nn.Sequential(
            nn.Linear(
                latent_dim,
                hidden_size,
                device=device,
            ),
            nn.Tanh(),
        )

        self.output_layer = nn.Sequential(
            nn.Linear(
                hidden_size,
                output_dim,
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

        return (
            output,
            self.get_internal_state(),
        )


# ── Recognition ───────────────────────────────────────────────────────────────


class RecLayer(nn.Module):
    def __init__(
        self,
        input_dim=1,
        latent_dim=1,
        device=None,
    ):
        super().__init__()

        self.latent_dim = latent_dim

        self.mean = nn.Sequential(
            nn.Linear(
                input_dim,
                latent_dim,
                device=device,
            ),
            nn.Tanh(),
        )

        self.log_var = nn.Sequential(
            nn.Linear(
                input_dim,
                latent_dim,
                device=device,
            )
        )

    def forward(
        self,
        x,
    ):

        mean = self.mean(x)

        # diagonal covariance
        std = torch.exp(0.5 * self.log_var(x))

        eps = torch.randn_like(std)

        z = mean + eps * std

        # Store diagonal covariance matrix
        R = torch.diag_embed(std)

        return mean, R, z


class Recognition(nn.Module):
    def __init__(
        self,
        input_dim=1,
        latent_dim=1,
        layers=1,
        device=None,
    ):
        super().__init__()

        self.rec_layers = nn.ModuleList(
            [
                RecLayer(
                    input_dim,
                    latent_dim,
                    device,
                )
                for _ in range(layers + 1)
            ]
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
        input_dim=1,
        hidden_size=1,
        latent_dim=1,
        output_dim=1,
        layers=1,
        seq_len=1,
        device=None,
    ):
        super().__init__()

        self.model_t = TimeRecognition(
            input_dim,
            hidden_size,
            seq_len,
            layers,
            device,
        )

        self.model_g = Generator(
            hidden_size,
            latent_dim,
            output_dim,
            layers,
            seq_len,
            device,
        )

        self.model_r = Recognition(
            input_dim,
            latent_dim,
            layers,
            device,
        )

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
        prediction,
        mean,
        R,
        generated_state,
        target_state,
    ):

        reconstruction = self.mse(
            prediction,
            y.reshape_as(prediction),
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

        prediction, state = self.model_g(x.size(0))

        loss = self.compute_loss(
            y,
            prediction,
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

        self.model_g.set_xi(z)

        prediction, state = self.model_g(x.size(0))

        return self.compute_loss(
            y,
            prediction,
            mean,
            R,
            state,
            t_1,
        ).item()


# ── Self Test ────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    from torch.optim import Adam

    torch.manual_seed(42)

    model = tDLGM(
        input_dim=10,
        hidden_size=20,
        latent_dim=5,
        output_dim=10,
        layers=2,
        seq_len=3,
        device=device,
    ).to(device)

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
