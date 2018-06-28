from header import *

class encoder(torch.nn.Module):

    def __init__(self, params):
        
        super(encoder, self).__init__()
        
        self.l0 = nn.Linear(params.X_dim, 3*params.h_dim, bias=True)
        self.l1 = nn.ReLU()
        self.bn = nn.BatchNorm1d(3*params.h_dim)
        
        self.l2 = nn.Linear(3*params.h_dim, 2*params.h_dim, bias=True)
        self.l3 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(2*params.h_dim)
        
        self.l4 = nn.Linear(2*params.h_dim, params.h_dim, bias=True)
        self.l5 = nn.ReLU()
        self.bn3 = nn.BatchNorm1d(params.h_dim)

        self.mu = nn.Linear(params.h_dim, params.Z_dim, bias=True)
        self.var = nn.Linear(params.h_dim, params.Z_dim, bias=True)
        
        torch.nn.init.xavier_uniform_(self.l0.weight)
        torch.nn.init.xavier_uniform_(self.l2.weight)
        torch.nn.init.xavier_uniform_(self.l4.weight)
        torch.nn.init.xavier_uniform_(self.mu.weight)
        torch.nn.init.xavier_uniform_(self.var.weight)

        
    def forward(self, inputs):
        
        o0 = self.l0(inputs)
        o1 = self.l1(o0)
        obn = self.bn(o1)

        o2 = self.l2(obn)
        o3 = self.l3(o2)
        obn2 = self.bn2(o3)
        
        o4 = self.l4(obn2)
        o5 = self.l5(o4)
        obn3 = self.bn3(o5)
        
        o6 = self.mu(obn3)
        o6_ = self.var(obn3)
        
        return o6, o6_
