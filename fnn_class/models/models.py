from header import *
import pdb
from weights_init import weights_init

class encoder(torch.nn.Module):

    def __init__(self, params):
        
        super(encoder, self).__init__()
        self.l0 = nn.Linear(params.X_dim, params.H_dim, bias=True)
        self.relu = nn.ReLU()
        self.drp_5 = nn.Dropout(.5)
        self.drp_1 = nn.Dropout(.1)
        self.bn = nn.BatchNorm1d(params.H_dim)
        self.bn_1 = nn.BatchNorm1d(params.h_dim)
        self.l2 = nn.Linear(params.H_dim, params.h_dim, bias=True)

        weights_init(self.l0.weight)
        weights_init(self.l2.weight)
        
    def forward(self, inputs):
        
        o = self.drp_5(inputs)
        o = self.l0(o)
        # o = self.bn(o)
        o = self.relu(o)
        o = self.drp_1(o)
        # ---------------------------------------
        o = self.l2(o)
        o = self.bn_1(o)
        o = self.relu(o)
        # o = self.drp_1(o)
        #----------------------------------------
	return o

# class variational(torch.nn.Module):
    
#     def __init__(self, params):
        
#         super(variational, self).__init__()
#         self.mu = nn.Linear(params.h_dim, params.Z_dim, bias=True)
#         self.var = nn.Linear(params.h_dim, params.Z_dim, bias=True)

#         weights_init(self.mu.weight)
#         weights_init(self.var.weight)
        
#     def forward(self, o):
        
#         o_ = self.mu(o)
#         o__ = self.var(o)
#         return o_, o__

class variational(torch.nn.Module):
    
    def __init__(self, params):
        
        super(variational, self).__init__()
        self.l0 = nn.Linear(params.X_dim, params.H_dim, bias=True)
        self.relu = nn.ReLU()
        self.drp_5 = nn.Dropout(.5)
        self.drp_1 = nn.Dropout(.1)
        self.bn = nn.BatchNorm1d(params.H_dim)
        self.bn_1 = nn.BatchNorm1d(params.h_dim)
        self.l2 = nn.Linear(params.H_dim, params.h_dim, bias=True)
        # ---------------------------------------------------------------
        self.mu = nn.Linear(params.h_dim, params.Z_dim, bias=True)
        self.var = nn.Linear(params.h_dim, params.Z_dim, bias=True)

        weights_init(self.l0.weight)
        weights_init(self.l2.weight)
        weights_init(self.mu.weight)
        weights_init(self.var.weight)
        
    def forward(self, inputs):
        
        o = self.drp_5(inputs)
        o = self.l0(o)
        # o = self.bn(o)
        o = self.relu(o)
        o = self.drp_1(o)
        # ---------------------------------------
        o = self.l2(o)
        o = self.bn_1(o)
        o = self.relu(o)
        # o = self.drp_1(o)
        # ---------------------------------------
        # ---------------------------------------
        o_ = self.mu(o)
        o__ = self.var(o)
        return o_, o__

class classifier(torch.nn.Module):
    
    def __init__(self, params):
        super(classifier, self).__init__()

        self.l0 = nn.Linear(params.X_dim, params.H_dim, bias=True)
        self.relu = nn.ReLU()
        self.drp_5 = nn.Dropout(.5)
        self.drp_4 = nn.Dropout(.4)
        self.drp_1 = nn.Dropout(.1)
        self.bn = nn.BatchNorm1d(params.H_dim)
        self.bn_1 = nn.BatchNorm1d(params.h_dim)
        self.l1 = nn.Linear(params.H_dim, params.h_dim, bias=True)
        # ---------------------------------------------------------------        
        self.l2 = nn.Linear(params.h_dim, params.y_dim, bias=True)
        self.bn = nn.BatchNorm1d(params.y_dim)
        
        weights_init(self.l0.weight)
        weights_init(self.l1.weight)
        weights_init(self.l2.weight)

    def forward(self, inputs):
        
        o = self.drp_5(inputs)
        o = self.l0(o)
        # o = self.bn(o)
        o = self.relu(o)
        o = self.drp_1(o)
        # ---------------------------------------
        o = self.l1(o)
        # o = self.bn_1(o)
        o = self.relu(o)
        # o = self.drp_1(o)
        # ---------------------------------------
        # ---------------------------------------
        o = self.drp_1(o) # added
        o = self.l2(o)

        # if(type(self.l3)!=str):
        return o




# class classifier(torch.nn.Module):

#     def __init__(self, params):
#         super(classifier, self).__init__()
#         self.l0 = nn.Linear(params.h_dim, params.y_dim, bias=True)
#         self.bn = nn.BatchNorm1d(params.y_dim)
#         self.drp_1 = nn.Dropout(.1)
        
#         weights_init(self.l0.weight)
        
#         if(params.fin_layer == "Sigmoid"):
#             self.l3 = nn.Sigmoid()
#         elif(params.fin_layer == "ReLU"):
#             self.l3 = nn.ReLU()
#         elif(params.fin_layer == "None"):
#             self.l3 = ""

#     def forward(self, o):
        
#         o = self.drp_1(o) # added
#         o = self.l0(o)
#         # o = self.drp(o)
#         # o = self.bn(o)

#         if(type(self.l3)!=str):
#             o = self.l3(o)

#         return o



