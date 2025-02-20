import torch.nn as nn
import torch

class Layer(nn.Module):

    def __init__(self, hidden_size=1, latent_dim=1, seq_len=1, device=None):
        super().__init__()
        self.hidden_size=hidden_size
        self.latent_dim=latent_dim
        self.seq_len=seq_len
        self.device=device

        self.t = nn.Sequential(
            nn.Linear(in_features=self.hidden_size,
                            out_features=self.hidden_size,device=device),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size,
                      out_features=self.hidden_size, device=device),
            nn.ReLU()
        ).to(self.device)
            
        
        self.g = nn.Sequential(
            nn.Linear(in_features=self.latent_dim,
                            out_features=self.latent_dim,
                            device=self.device),
            nn.Linear(in_features=self.latent_dim,
                            out_features=self.hidden_size,
                            device=self.device),
            nn.LeakyReLU()).to(self.device)


    def get_internal_state(self):
        return self.internal_state

    # Adding the noise and previous layer
    def forward(self, h, xi):
        h = self.t(h)
        return h + self.g(xi)

class Generator(nn.Module):

    def __init__(self, hidden_size=1, latent_dim=1, output_dim=1, layers=1, seq_len=1, device=None):
        super().__init__()
        self.output_dim = output_dim
        self.layers=layers
        self.hidden_size = hidden_size
        self.latent_dim=latent_dim
        self.seq_len=seq_len
        self.device=device
        self.make_network()
        self.xi = None

    def make_network(self):

        self.h_l = nn.ModuleList()

        for i in range(self.layers):
            self.h_l.append(Layer(self.hidden_size, self.latent_dim,self.seq_len, self.device))

        self.H_L = nn.Sequential(
            nn.Linear(in_features=self.latent_dim,
                            out_features=self.latent_dim,
                            device=self.device),
            nn.Linear(in_features=self.latent_dim,
                            out_features=self.hidden_size,
                            device=self.device),
            nn.Tanh()).to(self.device)

        self.h_0 = nn.Sequential(
                nn.Linear(in_features=self.hidden_size, out_features=self.output_dim, device=self.device),
                nn.Sigmoid()).to(self.device)

    def forward(self, batch_size=1):
        if self.xi is None:
            self.make_xi(batch_size)

        v = self.H_L(self.xi[0])
        count = 1
        for h in self.h_l:
            v = h(v, self.xi[count])
            count += 1
        return self.h_0(v)

    def set_xi(self, xi):
        self.xi = xi

    def make_xi(self, batch_size=1):
        self.xi = []
        for i in range(self.layers + 1):
            self.xi.append(torch.normal(mean=torch.zeros(batch_size, self.seq_len, self.latent_dim).to(self.device), std=1)
                           .to(self.device))




