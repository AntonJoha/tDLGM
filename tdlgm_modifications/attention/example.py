import torch
import torch.nn as nn
from recognition import Recognition
from generator import Generator
from time_recognition import TimeRecognition
import torch.nn.functional as F




def loss(x, x_hat, mean, R, mean_state, R_state, device=None, seq_len=1):
    
    mse = nn.MSELoss().to(device)
    l = F.binary_cross_entropy(x_hat, x, reduction='sum')
    amount = mean[0].size()[0]*mean[0].size()[1]
    for m, r in zip(mean, R):

        C = r @ r.transpose(-2,-1) 
        det = C.det() 
        l += 0.5 * torch.sum(m.pow(2).sum(-1) 
                             + C.diagonal(dim1=-2,dim2=-1).sum(-1)
                            -det.log()  -1)/amount

    amount = len(mean_state)*mean_state[0].size()[0]
    print(amount)
    for m_l, r_l in zip(mean_state, R_state):
        for m, r in zip(m_l, r_l):
            
            C = r @ r.transpose(-1, 0)
            det = C.det()
       
            l += 0.5* (
            torch.sum(m.pow(2).sum(-1) + C.diagonal(dim1=-2, dim2=-1).sum(-1) - det.log() - 1)
            )/amount
        
        
    
    #print(l, F.binary_cross_entropy(x_hat, x, reduction='sum'))
    return l 




batch_size = [4,3,1]
nn_size = [10,10]
input_dim = 4
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


mean_state, log_var, state = time(state_data)
print("LOG_VAR", log_var)
mean, R, z= m(stateless_data)

tDLGM.set_xi(z)
res = tDLGM(state)


loss(stateless_data, res, mean, R, mean_state, log_var)
print(res)


