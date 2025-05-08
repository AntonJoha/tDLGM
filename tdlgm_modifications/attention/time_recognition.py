from typing import List
from generator import Generator
import torch
import torch.nn as nn
import sys
from torch.autograd import Variable


# TODO ADD POSITIONAL ENCODING
# TODO Encoding should produce mean and variance

class TimeLayer(nn.Module):


    def __init__(self, batch_dim:int=1, input_dim:int=1, state_dim:int=1,network_size:List[int]=[1], activation_function=nn.LeakyReLU, device=None):
        super().__init__()

        self.batch_dim = batch_dim
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.activation_function = activation_function
        self.device = device
        self.network_input = self.input_dim*self.batch_dim
        self.network_size = network_size
        self.make_network()

    def make_network(self):


        network = [self.network_input]
        for i in self.network_size:
            network.append(i)

        layers = []
        for i in range(len(network)-1):
            layers.append(nn.Linear(in_features=network[i],out_features=network[i+1]))
            layers.append(self.activation_function())
        layers.append(nn.Linear(in_features=network[-1], out_features=self.state_dim))

        self.mean = nn.Sequential(*layers)
        self.var = nn.Sequential(*layers)
    

    def forward(self, x):
        log_var = self.var(x)
        mean = self.mean(x)
        return mean, log_var, mean + self.reparameterization(log_var)


    def reparameterization(self,log_var):
        dims = log_var.size()
        eps = Variable(torch.FloatTensor(dims).normal_())
        std = log_var.mul(0.5).exp_()
        return eps.mul(std)


class TimeRecognition(nn.Module):


    def __init__(self, input_dim:int=1, network_size:List[int]=[1], batches=[1],seq_len=1, state_dim=1, device=None):
        super().__init__()

        
        self.state_dim = state_dim
        self.network_size = network_size
        self.batches = batches
        self.input_dim = input_dim
        self.seq_len=seq_len

        self.network = []
        self.make_network()

    def make_network(self):
        self.network = []
        for b in self.batches:
            self.network.append(TimeLayer(batch_dim=b, input_dim=self.input_dim,state_dim=self.state_dim,network_size=self.network_size))


    def get_data(self, seq, batches):
        p = list(torch.split(seq, batches, dim=1)[-self.seq_len:])
        #padding
        entries = p[0].shape[1]
        if p[-1].shape[1] != entries:
            p[-1] = torch.cat((p[-1], torch.zeros(p[-1].shape[0], entries - p[-1].shape[1], p[-1].shape[2])), dim=1)

        res = torch.stack(p, dim=1) # Shape: (batch_size, seq_len, time_steps, features)
        

        res = res.reshape(res.shape[0], res.shape[1], -1) # Shape: (batch_size, seq_len, time_steps*features)
        return res


    def forward(self, x):
        res = []
        mean = []
        r = []
        z = []
        for b, n in zip(self.batches, self.network):
            m, r_curr, z_curr = n(self.get_data(x,b))
            mean.append(m)
            r.append(r_curr)
            z.append(z_curr)
        

        return mean, r, z
    




if __name__ == "__main__":

#[
    m = TimeRecognition(input_dim=2, network_size=[10,10], batches=[2,1], seq_len=3, state_dim=10)
    seq = torch.rand((10,2,2))
    res = m(seq)
    print(res)
    tDLGM = Generator(state_dim=10,
                      network_size=[10,10],
                      latent_dim=1,
                      layers=2,
                      num_heads=2,
                      output_dim=3,
                      seq_len=3,
                      activation_function=nn.Sigmoid,
                      device=None)
    

    r = tDLGM(res[2], batch_size=2)
    print("Result: ", r.shape)
