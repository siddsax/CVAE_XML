from header import *
import pdb
from weights_init import weights_init

class encoder(torch.nn.Module):

    def __init__(self, params):
        
        super(encoder, self).__init__()
        self.l0 = nn.Linear(params.X_dim, params.h_dim, bias=True)
        self.relu = nn.ReLU()
        self.drp = nn.Dropout(.5)
        self.bn = nn.BatchNorm1d(params.h_dim)
        self.l2 = nn.Linear(params.h_dim, params.H_dim, bias=True)
        # self.mu = nn.Linear(params.h_dim, params.Z_dim, bias=True)

        weights_init(self.l0.weight)
        weights_init(self.l2.weight)
        
    def forward(self, inputs):
        
        o = self.drp(inputs)
        o = self.l0(o)
        o = self.relu(o)
        o = self.drp(o)
        o = self.bn(o)
        o = self.l2(o)
        o = self.relu(o)
        return o

class variational(torch.nn.Module):
    
    def __init__(self, params):
        
        super(variational, self).__init__()
        self.mu = nn.Linear(params.H_dim, params.Z_dim, bias=True)
        self.var = nn.Linear(params.H_dim, params.Z_dim, bias=True)

        weights_init(self.mu.weight)
        weights_init(self.var.weight)
        
    def forward(self, o):
        
        o_ = self.mu(o)
        o__ = self.var(o)
        return o_, o__

class classifier(torch.nn.Module):

    def __init__(self, params):
        super(classifier, self).__init__()
        # self.l0 = nn.Linear(params.h_dim, params.H_dim, bias=True)
        self.l0 = nn.Linear(params.H_dim, params.y_dim, bias=True)
        self.bn = nn.BatchNorm1d(params.y_dim)
        self.drp = nn.Dropout(.5)
        
        weights_init(self.l0.weight)
        # weights_init(self.l1.weight)
        
        if(params.fin_layer == "Sigmoid"):
            self.l3 = nn.Sigmoid()
        elif(params.fin_layer == "ReLU"):
            self.l3 = nn.ReLU()
        elif(params.fin_layer == "None"):
            self.l3 = ""

    def forward(self, o):
        
        o = self.l0(o)
        # o = self.drp(o)
        # o = self.bn(o)
        # o = self.l1(o)

        if(type(self.l3)!=str):
            o = self.l3(o)

        return o
