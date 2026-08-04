from dataclasses import dataclass
from itertools import chain

import torch
from torch import nn


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Configuration
# ============================================================


@dataclass(slots=True)
class tDLGMConfig:

    # Architecture
    input_dim: int = 10
    hidden_size: int = 20
    output_dim: int = 10
    layers: int = 2

    # Training
    learning_rate: float = 1e-3
    batch_size: int = 64

    # Misc
    seed: int = 42
    device: str | None = None



# ============================================================
# Time Recognition
# ============================================================


class TimeLayer(nn.Module):

    """
    Encodes observed history into a temporal state.

    This is the only recognition model.
    It receives only past observations.
    """


    def __init__(
        self,
        input_dim,
        hidden_size,
        device=None,
    ):

        super().__init__()

        self.hidden_size = hidden_size


        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            batch_first=True,
            device=device,
        )


    def forward(self,x):

        _,(h,c)=self.lstm(x)

        return (
            h.squeeze(0),
            c.squeeze(0),
        )



class TimeRecognition(nn.Module):

    def __init__(
        self,
        config,
    ):

        super().__init__()

        self.time_layers = nn.ModuleList(
            [
                TimeLayer(
                    config.input_dim,
                    config.hidden_size,
                    device,
                )
                for _ in range(config.layers)
            ]
        )


    def forward(self,x):

        return [
            layer(x)
            for layer in self.time_layers
        ]



# ============================================================
# Generator
# ============================================================


class GenLayer(nn.Module):


    def __init__(
        self,
        hidden_size,
        device=None,
    ):

        super().__init__()

        self.hidden_size = hidden_size


        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            device=device,
        )


    def forward(
        self,
        x,
        state,
    ):

        h,c = state

        h = h.unsqueeze(0)
        c = c.unsqueeze(0)


        output,(h,c)=self.lstm(
            x,
            (h,c),
        )


        return output,(
            h.squeeze(0),
            c.squeeze(0),
        )




class Generator(nn.Module):


    def __init__(
        self,
        config,
    ):

        super().__init__()


        self.layers = nn.ModuleList(
            [
                GenLayer(
                    config.hidden_size,
                    device,
                )
                for _ in range(config.layers)
            ]
        )


        self.output_layer = nn.Linear(
            config.hidden_size,
            2 * config.output_dim,
            device=device,
        )



    def forward(
        self,
        states,
    ):


        #
        # Start generation from temporal state
        #

        x = states[0][0].unsqueeze(1)


        next_states=[]


        for layer,state in zip(
            self.layers,
            states,
            strict=False,
        ):

            x,state = layer(
                x,
                state,
            )

            next_states.append(state)



        x=x[:,-1,:]


        output=self.output_layer(x)


        mean,log_var=output.chunk(
            2,
            dim=-1,
        )


        log_var=log_var.clamp(
            -5,
            5,
        )


        return (
            mean,
            log_var,
            next_states,
        )



# ============================================================
# tDLGM
# ============================================================


class tDLGM(nn.Module):


    def __init__(
        self,
        config,
    ):

        super().__init__()

        self.config=config


        self.model_t=TimeRecognition(
            config
        )


        self.model_g=Generator(
            config
        )


        self.loss=nn.GaussianNLLLoss()



    def get_parameters(self):

        return chain(
            self.model_t.parameters(),
            self.model_g.parameters(),
        )



    def forward(
        self,
        x,
    ):

        state=self.model_t(x)


        mean,log_var,_=self.model_g(
            state
        )


        return (
            mean,
            log_var,
        )



    def compute_loss(
        self,
        x,
        y,
    ):


        pred_mean,pred_log_var=self.forward(
            x
        )


        # support [batch,horizon,dim]
        # by predicting final horizon step

        if y.ndim == 3:
            y=y[:,-1,:]


        loss=self.loss(
            pred_mean,
            y,
            pred_log_var.exp(),
        )


        return loss



    def train_step(
        self,
        x,
        y,
        optimizer,
    ):

        self.train()


        optimizer.zero_grad()


        loss=self.compute_loss(
            x,
            y,
        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(
            self.parameters(),
            5.0,
        )


        optimizer.step()


        return loss.item()
    def _compute_losses(
        self,
        x,
        y,
    ):

        pred_mean, pred_log_var = self.forward(x)


        if y.ndim == 3:
            y = y[:, -1, :]


        reconstruction = (
            0.5
            *
            (
                pred_log_var
                +
                (y - pred_mean).pow(2)
                /
                pred_log_var.exp()
            )
        ).mean()


        # No latent variables exist anymore
        # therefore no KL term

        kl = torch.tensor(
            0.0,
            device=x.device,
        )


        return reconstruction, kl



    @torch.no_grad()
    def compute_losses(
        self,
        x,
        y,
    ):

        self.eval()


        reconstruction, kl = self._compute_losses(
            x,
            y,
        )


        return (
            reconstruction.item(),
            kl.item(),
        )



    @torch.no_grad()
    def get_loss(
        self,
        x,
        y,
    ):

        self.eval()


        loss = self.compute_loss(
            x,
            y,
        )


        return loss.item()



# ============================================================
# Self test
# ============================================================


def main():

    from torch.optim import Adam


    torch.manual_seed(42)


    config=tDLGMConfig()


    model=tDLGM(
        config
    ).to(device)



    optimizer=Adam(
        model.get_parameters(),
        lr=1e-3,
    )


    # --------------------------------------------------------
    # Synthetic forecasting problem
    #
    # x_t -> x_(t+1)
    #
    # --------------------------------------------------------


    batch_size=64
    seq_len=10
    input_dim=10


    x=torch.randn(
        batch_size,
        seq_len,
        input_dim,
        device=device,
    )


    y=torch.randn(
        batch_size,
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


    # --------------------------------------------------------
    # Before training
    # --------------------------------------------------------


    before=model.get_loss(
        x,
        y,
    )


    print(
        f"Loss before training: {before:.5f}"
    )


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------


    model.train()


    for step in range(300):

        loss=model.train_step(
            x,
            y,
            optimizer,
        )


        if step % 50 == 0:

            print(
                f"Step {step}: {loss:.5f}"
            )


    # --------------------------------------------------------
    # After training
    # --------------------------------------------------------


    after=model.get_loss(
        x,
        y,
    )


    print(
        f"Loss after training: {after:.5f}"
    )


    assert torch.isfinite(
        torch.tensor(after)
    )


    assert after < before, (
        "Model did not improve"
    )


    print(
        "Test passed."
    )



if __name__ == "__main__":

    main()
