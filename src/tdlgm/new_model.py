import torch
from torch import nn

from experiments.util import SeriesConfig

TDLGMConfig = SeriesConfig


def _resolve_num_heads(hidden_dim: int) -> int:
    for candidate in (8, 4, 2, 1):
        if hidden_dim % candidate == 0 and hidden_dim // candidate >= 16:
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

        self.kl_multiplier = 1
        self.beta = config.beta

        self.prior_state = SequenceAttentionEncoder(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            layers=config.layers,
            seq_len=config.seq_len,
        )
        self.posterior_state = SequenceAttentionEncoder(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            layers=config.layers,
            seq_len=config.seq_len + config.horizon,
        )

        self.input_prior = _make_mlp(
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim*2*config.horizon,
        )
        self.input_posterior = _make_mlp(
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim*2*config.horizon,
        )
        self.config = config
        
        self.nllLoss = nn.GaussianNLLLoss()
        self.model_layers = self._make_tdlgm_layers(config)

        self.beta_max = config.beta
        self.epoch = 1
    

    def _make_tdlgm_layers(self, config: TDLGMConfig) -> nn.ModuleList:
        layers = nn.ModuleList()
        for _ in range(config.layers):
            layers.append(self._make_tdlgm_layer(config))
                    
        return layers

    def _make_tdlgm_layer(self, config: TDLGMConfig) -> nn.ModuleList:
        return nn.ModuleDict(
                {
                    "prior": _make_mlp(
                        input_dim=config.hidden_dim,
                        hidden_dim=config.hidden_dim,
                        output_dim=config.output_dim*2*config.horizon,
                    ),
                    "posterior": _make_mlp(
                        input_dim=config.hidden_dim,
                        hidden_dim=config.hidden_dim,
                        output_dim=config.output_dim*2*config.horizon,
                        ),
                    "generator": _make_mlp(
                        input_dim=config.output_dim*config.horizon,
                        hidden_dim=config.hidden_dim,
                        output_dim=config.output_dim*2*config.horizon,
                    ),
                    }
                )


    def _reparametrize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std



    def _to_output_shape(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), self.config.horizon, self.config.output_dim)
        return x.squeeze(-1) if self.config.output_dim == 1 else x


    
    def _multiply_gaussians(self, mean1: torch.Tensor, logvar1: torch.Tensor, mean2: torch.Tensor, logvar2: torch.Tensor):
        # https://ccrma.stanford.edu/~jos/sasp/Product_Two_Gaussian_PDFs.html
        var1 = torch.exp(logvar1)
        var2 = torch.exp(logvar2)

        combined_var = 1 / (1 / var1 + 1 / var2)
        combined_mean = combined_var * (mean1 / var1 + mean2 / var2)

        combined_logvar = torch.log(combined_var)
        return combined_mean, combined_logvar

    def _state_modification(self, state, layer_index):
        if layer_index == 0:
            return state.mean(dim=-2) # FIRST LAYER? ?? ?  ?  ??
        else:
            return state.mean(dim=-2) # OTHER LAYERS? ?? ?  ?  ??



    def _get_state(self, x: torch.Tensor, prior=True) -> torch.Tensor:
        if prior:
            prior_state = self.prior_state(x)
            return prior_state

        else:
            posterior_state = self.posterior_state(x)
            return posterior_state
    

    def _split_mean_logvar(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, logvar = h.chunk(2, dim=-1)
        return mean, logvar


    def _latent_pass(self, x, y=None, prior=True):

        # only used for training, not for inference
        if y is not None:
            mean_q_list, logvar_q_list = [], []
            mean_p_list, logvar_p_list = [], []
        else:
            mean_q_list, logvar_q_list = None, None
            mean_p_list, logvar_p_list = None, None
       

        y_full = None
        if y is not None:
            y_full = torch.cat([x, y], dim=-2)  # this is what the posterior models use for training

        q_raw_state = None

        p_raw_state = self._get_state(x)
        if y is not None:
            q_raw_state = self._get_state(y_full, prior=False)

        hL_q = None

        hL_p = self.input_prior(self._state_modification(p_raw_state, layer_index=0))
        mean_p, logvar_p = self._split_mean_logvar(hL_p)
        if y is not None:
            hL_q = self.input_posterior(self._state_modification(q_raw_state, layer_index=0))

            mean_q, logvar_q = self._split_mean_logvar(hL_q)
            mean_q_list.append(mean_q)
            logvar_q_list.append(logvar_q)

            mean_p_list.append(mean_p)
            logvar_p_list.append(logvar_p)

        if prior:
            mean_l, logvar_l = self._split_mean_logvar(hL_p)
        else:
            mean_l, logvar_l = self._split_mean_logvar(hL_q)

        for layer_index, layer in enumerate(self.model_layers):
        
            q_state = None
            if y is not None:
                q_state = self._state_modification(q_raw_state, layer_index)
            p_state = self._state_modification(p_raw_state, layer_index)

            mean_s_p, logvar_s_p = self._split_mean_logvar(layer["prior"](p_state))
            mean_s_q, logvar_s_q = None, None
            if y is not None:
                mean_s_q, logvar_s_q = self._split_mean_logvar(layer["posterior"](q_state))
                mean_q_list.append(mean_s_q)
                logvar_q_list.append(logvar_s_q)
                mean_p_list.append(mean_s_p)
                logvar_p_list.append(logvar_s_p)


            if prior:
                mean_s, logvar_s = mean_s_p, logvar_s_p
            else:
                mean_s, logvar_s = mean_s_q, logvar_s_q


                
            mean, logvar = self._multiply_gaussians(mean_l, logvar_l, mean_s, logvar_s)
            
            z = self._reparametrize(mean, logvar)
            mean_l, logvar_l = self._split_mean_logvar(layer["generator"](z))
        
        pred_mean = self._to_output_shape(mean_l)
        pred_logvar = self._to_output_shape(logvar_l)
        return pred_mean, pred_logvar, mean_q_list, logvar_q_list, mean_p_list, logvar_p_list
    
    def train_step(self, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:

        self.train()

        self.kl_multiplier = self.beta_max * min(1.0, (self.epoch + 1) / 10)
        optimizer.zero_grad()
        (pred_mean,pred_logvar, mean_q_list,logvar_q_list,mean_p_list,logvar_p_list,) = self._latent_pass(x, y, prior=False)
        loss = self._compute_loss(
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


    def _gaussian_kl(
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



    def _compute_losses(self, y, pred_mean, pred_logvar, mean_q_list, logvar_q_list, mean_p_list, logvar_p_list):
        # Reconstruction loss
        recon_loss = self.nllLoss(pred_mean, y.squeeze(-1), pred_logvar.exp())

        # KL divergence loss
        kl_loss = 0.0
        for mean_q, logvar_q, mean_p, logvar_p in zip(mean_q_list, logvar_q_list, mean_p_list, logvar_p_list):
            kl_loss += self._gaussian_kl(mean_q, logvar_q, mean_p, logvar_p)

        return recon_loss, kl_loss * self.kl_multiplier

    def _compute_loss(self, y, pred_mean, pred_logvar, mean_q_list, logvar_q_list, mean_p_list, logvar_p_list):

        # Reconstruction loss        total_loss = recon_loss + self.kl_multiplier * kl_loss
        recon_loss, kl_loss = self._compute_losses(
            y, pred_mean, pred_logvar, mean_q_list, logvar_q_list, mean_p_list, logvar_p_list
        )
        return recon_loss + kl_loss
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean, logvar, *_  = self._latent_pass(x, None, prior=True)
        return mean, logvar
    

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


