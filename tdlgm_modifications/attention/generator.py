from typing import List
import torch.nn as nn
import torch

class Layer(nn.Module):

    def __init__(self, network_size: List[int]=[1],
                 state_dim:int=1,
                 latent_dim:int =1,
                 device=None,
                 num_heads:int=2,
                 seq_len:int=2,
                 activation_function=nn.LeakyReLU):
        super().__init__()

        self.device = device
        self.latent_dim = latent_dim
        self.network_size = network_size
        self.activation_function = activation_function
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.state_dim = state_dim

        # Make neural networks
        self.make_attention()
        self.make_layers()



    # Define attention function and function to make key and query
    def make_attention(self):
        self.attention = nn.MultiheadAttention(self.state_dim, self.num_heads)
        self.make_key = nn.Sequential(
                    nn.Linear(in_features=self.state_dim, out_features=self.state_dim)
                )
        self.make_query = nn.Sequential(
                    nn.Linear(in_features=self.state_dim, out_features=self.state_dim)
                )

    # Make the neural network and the "matrix" G
    def make_layers(self):
        self.g = nn.Sequential(
                nn.Linear(in_features=self.latent_dim,
                          out_features=self.latent_dim),
                nn.Linear(in_features=self.latent_dim,
                          out_features=self.latent_dim),
                )

        in_dim = self.latent_dim + self.state_dim*self.seq_len
        network = [in_dim]
        for i in self.network_size:
            network.append(i)

        layers = []
        for i in range(len(network)-1):
            layers.append(nn.Linear(in_features=network[i],out_features=network[i+1]))
            layers.append(self.activation_function())
        layers.append(nn.Linear(in_features=network[-1], out_features=self.latent_dim))

        self.ffn = nn.Sequential(*layers)

    # Adding the noise and previous layer
    def forward(self, val, h, xi):
        key = self.make_key(val)
        query = self.make_query(val)
        val = val
        
        att, _ = self.attention(query, key, val)
        res = att.transpose(0,1).reshape(att.shape[1], att.shape[0]*att.shape[2])
        in_data = torch.cat((res,h),dim=-1)

        return self.ffn(in_data) + xi



class Generator(nn.Module):

    def __init__(self, state_dim: int=1, network_size: List[int] =[1], latent_dim=1, output_dim=1, layers=1,num_heads=1, seq_len=1,activation_function=nn.LeakyReLU,device=None):
        super().__init__()
        
        self.state_dim = state_dim # Size of the state space
        self.network_size = network_size # Vector over the network
        self.num_heads=num_heads
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.layers = layers
        self.activation_function = activation_function
        self.device = device
        self.h_l = []
        self.xi = None
        
        self.make_network()


    

    def make_network(self):
        for _ in range(self.layers):
            self.h_l.append(Layer(network_size=self.network_size,
                                      state_dim=self.state_dim,
                                      latent_dim=self.latent_dim,
                                      device=self.device,
                                      num_heads=self.num_heads,
                                        seq_len=self.seq_len,
                                      activation_function=self.activation_function).to(self.device))


        self.H_L = nn.Sequential(
            nn.Linear(in_features=self.latent_dim,
                            out_features=self.latent_dim,
                            device=self.device),
            nn.Linear(in_features=self.latent_dim,
                            out_features=self.latent_dim,
                            device=self.device))

        self.h_0 = nn.Sequential(
                nn.Linear(in_features=self.latent_dim, out_features=self.output_dim, device=self.device),
                nn.Sigmoid()).to(self.device)


    def forward(self, state_space,batch_size=1):
        if self.xi is None:
            self.make_xi(batch_size)

        assert self.xi is not None
        print("XI SHAPE:", self.xi[0].shape)
        h_out = self.H_L(self.xi[0])

        for h, s, xi in zip(self.h_l, state_space, self.xi[1:]):
            h_out = h(s, h_out, xi)

        return self.h_0(h_out)


    def set_xi(self, xi):
        self.xi = xi

    def make_xi(self, batch_size=1):
        self.xi = []
        for _ in range(self.layers + 1):
            self.xi.append(torch.normal(mean=torch.zeros(batch_size,self.latent_dim).to(self.device), std=1)
                           .to(self.device))




if __name__ == "__main__":

    tDLGM = Generator(state_dim=10,
                      network_size=[10,10],
                      latent_dim=10,
                      layers=3,
                      num_heads=2,
                      output_dim=10,
                      seq_len=3,
                      activation_function=nn.Sigmoid,
                      device=None)
    
    b_s = 4
    state_space = torch.rand((3,3,b_s,10))
    # Layout of state space is (layer, seq_len, batch_size, values)
    print(tDLGM(state_space,batch_size=b_s))



