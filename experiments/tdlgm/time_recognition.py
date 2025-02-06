import torch
import torch.nn as nn


class TimeLayer(nn.Module):


    def __init__(self, input_dim=1, hidden_size=1, seq_len=1, device=None):
        super().__init__()
            
        self.input_dim=input_dim
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.device = device

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=self.hidden_size,num_layers=1, batch_first=True).to(self.device)
    
    def init_hidden(self, size):
        self.internal_state = [
                    torch.zeros(1, size[0], self.hidden_size,device=self.device).to(self.device),
                    torch.zeros(1, size[0], self.hidden_size,device=self.device).to(self.device)
                ]



    def forward(self, x):
        self.init_hidden(x.size())
        _, h = self.lstm(x)# , self.internal_state)
        return h

class TimeRecognition(nn.Module):



    def __init__(self, input_dim=1, hidden_size=1, seq_len=1, layers=1, device=None):
        super().__init__()

        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.layers = layers
        self.device = device
        self.input_dim=input_dim

        self.make_network()

    def forward(self, x):
        res = []
        for l in self.h:
            res.append(l(x))
        return res
    
    def make_network(self):

        self.h = nn.ModuleList()

        for i in range(self.layers):
            self.h.append(TimeLayer(input_dim=self.input_dim,
                                    hidden_size=self.hidden_size,
                                    seq_len=self.seq_len,
                                    device=self.device).to(self.device))

