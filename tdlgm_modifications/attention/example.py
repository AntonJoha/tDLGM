import torch
import torch.nn as nn
from recognition import Recognition
from generator import Generator
from time_recognition import TimeRecognition







batch_size = [4,3,1]
nn_size = [10,10]
input_dim = 10
latent_dim = 2
state_dim = 2
layers = 3
output_dim = 4
seq_len = 2


state_data = torch.rand((10,3,input_dim))
stateless_data =  torch.rand((3, input_dim))


tDLGM = Generator(state_dim=state_dim,
                  network_size=nn_size,
                  latent_dim=latent_dim,
                  layers=layers,
                  num_heads=2,
                  output_dim=output_dim,
                  seq_len=seq_len,
                  activation_function=nn.Sigmoid,
                  device=None)



m = Recognition(input_dim=input_dim, latent_dim=latent_dim,layers=layers)

time = TimeRecognition(input_dim=input_dim, network_size=nn_size, batches=batch_size, seq_len=seq_len, state_dim=state_dim)


mean, log_var, state = time(state_data)
mean, R, z= m(stateless_data)

tDLGM.set_xi(z)
res = tDLGM(state)

print(res)
