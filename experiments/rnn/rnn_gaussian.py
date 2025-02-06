import torch.nn as nn
import torch

class PredictTime(nn.Module):
    def __init__(self, input_size=1,output_size=1,hidden_layers=1,h1=1, h2=1,noise=False, device=None):

        super().__init__()
        

        self.device=device

        self.hidden_layers = hidden_layers
        self.input_size = input_size
        self.output_size = output_size
        self.h1 = h1
        self.h2 = h2
        self.noise = noise

        self.lout = None
        self.gkin = None
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=self.h1, num_layers=1, batch_first=True).to(device) # two lstm different hidden size
        if hidden_layers == 2:
            self.lstm2 = nn.LSTM(input_size=self.h1, hidden_size=self.h2, num_layers=1,batch_first=True).to(device)
            self.lout = nn.Linear(self.h2,self.output_size).to(device)
            self.gkin = nn.Linear(self.h1,self.h2).to(device)

        else:
            self.lout = nn.Linear(self.h1, self.output_size).to(device)
            self.gkin = nn.Linear(self.h1,self.h1).to(device)

        
        self.clean_state()
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
 
        x, self.in1 = self.lstm1(x, self.in1)
        
        y = None
        if self.noise:
            y = torch.randn(x.size()).to(self.device)
            y = self.gkin(y)
        if self.hidden_layers == 2:
            x, self.in2 = self.lstm2(x, self.in2)
        
        if self.noise:
            x = x + y
        x = self.lout(x)
        
        return self.sig(x[:,-1,:])
    
    def init_state(self, batch_size=1):
        self.h_0 = self.in1[0].detach()
        self.c_0 = self.in1[1].detach()
        self.in1 = (self.h_0, self.c_0)
        self.h_0 = self.in2[0].detach()
        self.c_0 = self.in2[1].detach()
        self.in2 = (self.h_0, self.c_0)
    
    def clean_state(self, batch_size=1):
        self.h_0 = torch.zeros(1,batch_size, self.h1, device=self.device)
        self.c_0 = torch.zeros(1, batch_size,  self.h1, device=self.device)
        self.in1 = (self.h_0, self.c_0)
        self.h_0 = torch.zeros(1,batch_size, self.h2, device=self.device)
        self.c_0 = torch.zeros(1, batch_size, self.h2, device=self.device)
        self.in2 = (self.h_0, self.c_0)



    def random_state(self, batch_size=1):
        self.h_0 = torch.randn(1,batch_size,self.h1, device=self.device)
        self.c_0 = torch.randn(1,batch_size,self.h1, device=self.device)
        self.in1 = (self.h_0, self.c_0)
        self.h_0 = torch.randn(1,batch_size,self.h2, device=self.device)
        self.c_0 = torch.randn(1,batch_size,self.h2, device=self.device)
        self.in2 = (self.h_0, self.c_0)


