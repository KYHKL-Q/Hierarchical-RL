import torch
from torch import nn,optim
import torch.nn.functional as F

class Anet(nn.Module):
    def __init__(self):
        super(Anet,self).__init__()
        self.cov1=nn.Conv1d(in_channels=6,out_channels=16,kernel_size=5,stride=1,padding=2,bias=True)
        self.cov2=nn.Conv1d(in_channels=16,out_channels=32,kernel_size=5,stride=1,padding=2,bias=True)
        self.cov3=nn.Conv1d(in_channels=32,out_channels=16,kernel_size=5,stride=2,padding=0,bias=True)
        self.cov4=nn.Conv1d(in_channels=16,out_channels=4,kernel_size=5,stride=2,padding=0,bias=True)
        self.lin1=nn.Linear(in_features=664,out_features=128,bias=True)
        self.lin2=nn.Linear(in_features=128,out_features=16,bias=True)
        self.lin3=nn.Linear(in_features=16,out_features=1,bias=True)

        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mask=torch.tensor([[0.01,0,0,0,0,0],
                                [0,0.05,0,0,0,0],
                                [0,0,0.01,0,0,0],
                                [0,0,0,0.05,0,0],
                                [0,0,0,0,0.01,0],
                                [0,0,0,0,0,0.05]],dtype=torch.float).to(self.device)

    def forward(self,state):
        x1 = torch.matmul(state, self.mask).permute(0, 2, 1)
        c1 = F.leaky_relu(self.cov1(x1), 0.2)
        c2 = F.leaky_relu(self.cov2(c1), 0.2)
        c3 = F.leaky_relu(self.cov3(c2), 0.2)
        c4 = F.leaky_relu(self.cov4(c3), 0.2)
        l1 = F.leaky_relu(self.lin1(c4.view(state.shape[0], -1)), 0.2)
        l2 = F.leaky_relu(self.lin2(l1), 0.2)
        l3 = torch.sigmoid(self.lin3(l2))

        return l3