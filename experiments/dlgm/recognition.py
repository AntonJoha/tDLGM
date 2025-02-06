import torch
import torch.nn as nn

class Layer(nn.Module):

    
    def __init__(self, input_dim=1, latent_dim=1, device=None):
        super().__init__()
        self.latent_dim=latent_dim
        self.input_dim=input_dim
        self.device = device

        self.d = nn.Sequential(
                nn.Linear(self.input_dim, self.latent_dim),
                nn.Sigmoid(),
                nn.Linear(self.latent_dim,self.latent_dim),
                nn.Sigmoid()).to(device)
        self.u = nn.Sequential(
                    nn.Linear(self.input_dim, self.latent_dim),
                    nn.Sigmoid(),
                    nn.Linear(self.latent_dim, self.latent_dim),
                    nn.Sigmoid()
                ).to(device)
        self.mean = nn.Sequential(
                    nn.Linear(self.input_dim, self.latent_dim),
                    nn.Tanh(),
                    nn.Linear(self.latent_dim, self.latent_dim),
                    nn.Tanh()
                ).to(device)

    def forward(self, x):
        d = self.d(x)
        u = self.u(x)
        mean = self.mean(x)
        R = self.calculate_r(d,u)
        z = self.calculate_z(mean,R)
        return mean, R, z

    
    def calculate_z(self, mean, R):
        v = torch.randn(mean.size()).unsqueeze(-1).to(self.device)
        mult = torch.matmul(R, v).squeeze()
        return mult + mean
    

    def calculate_r(self, d, u):
        D = torch.diag_embed(d)
        epsilon = 1e-6
        D_inv = torch.inverse(D)  + epsilon
        D_in_sqr = torch.sqrt(D_inv)
        u_r = u.unsqueeze(-1)
        U = torch.matmul(u_r, u_r.transpose(-2,-1))
        ut_d_inv_u = torch.matmul(u_r.transpose(-2,-1), torch.matmul(D_inv, u_r))
        eta = 1/(1 + ut_d_inv_u)
        right = (1 - torch.sqrt(eta)) / ut_d_inv_u
        R = D_in_sqr - right*torch.matmul(D_inv, torch.matmul(U, D_in_sqr))
        return R

class Recognition(nn.Module):


    def __init__(self, input_dim=1, latent_dim=1, layers=1, device=None):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.layers=layers+1
        self.device = device

        self.make_network()


    def make_network(self):

        self.g = nn.ModuleList()
        for i in range(self.layers):
            self.g.append(Layer(self.input_dim, self.latent_dim, self.device).to(self.device))


    def forward(self, x):
        R = []
        mean = []
        z = []
        for l in self.g:
            res = l(x)
            R.append(res[1])
            mean.append(res[0])
            z.append(res[2])
        
        return mean, R, z
