import torch
from torch import nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.input_size=673
        self.hidden_size=1024
        self.num_layers=4
        self.output_size=673

        self.rnn=nn.GRU(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers,batch_first=True)
        self.lin=nn.Linear(in_features=self.hidden_size,out_features=self.output_size)

    def forward(self,x,h0=0,is_eval=False):
        if not is_eval:
            out,h=self.rnn(x)
        else:
            out,h=self.rnn(x,h0)
        out = self.lin(out)

        return out,h