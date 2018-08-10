from header import *
import pdb
from weights_init import weights_init

class variational(torch.nn.Module):

    def __init__(self, params):

        super(variational, self).__init__()
        self.nrm = nn.BatchNorm1d(params.X_dim)
        # ---------------------------------------------------------------

        self.params = params
        # self.emb_layer = nn.Linear(params.y_dim, params.e_dim, bias=False) 
        # self.emb_layer.weight = torch.nn.Parameter(torch.from_numpy(params.w2v_w).type(params.dtype))
        # self.emb_layer.weight.requires_grad = False
        self.w2v_w = torch.from_numpy(params.w2v_w).type(params.dtype)
        if(self.params.layer_y):
            self.l0 = nn.Linear(params.X_dim + params.e_dim, params.H_dim, bias=True)
            self.bn_cat = nn.BatchNorm1d(params.X_dim + params.e_dim)
        else:
            self.l0 = nn.Linear(params.X_dim + params.y_dim, params.H_dim, bias=True)
            self.bn_cat = nn.BatchNorm1d(params.X_dim + params.y_dim)

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

    def forward(self, X, y, eps=1e-7):

        if(self.params.layer_y):
            y_sum = torch.sum(y, dim=1).view(-1,1) + eps
            # divisor = torch.max(y_sum, Variable(torch.ones(y_sum.shape).type(self.params.dtype)))
            y = torch.mm(y, Variable(self.w2v_w))
            y = y/y_sum
        o = torch.cat((y,X), dim=-1)
        o = self.bn_cat(o)
        # ---------------------------------------

        o = self.l0(o)
        o = self.relu(o)
        # ---------------------------------------
        o = self.l2(o)
        o = self.relu(o)
        # ---------------------------------------
        o_ = self.mu(o)
        o__ = self.var(o)
        return o_, o__

class variationalRes(torch.nn.Module):

    def __init__(self, params):

        super(variationalRes, self).__init__()
        self.nrm = nn.BatchNorm1d(params.X_dim)
        # ---------------------------------------------------------------

        self.params = params
        # self.emb_layer = nn.Linear(params.y_dim, params.e_dim, bias=False) 
        # self.emb_layer.weight = torch.nn.Parameter(torch.from_numpy(params.w2v_w).type(params.dtype))
        # self.emb_layer.weight.requires_grad = False
        self.w2v_w = torch.from_numpy(params.w2v_w).type(params.dtype)
        # if(self.params.layer_y):
        #     self.l0 = nn.Linear(params.X_dim + params.e_dim, params.H_dim, bias=True)
        #     self.bn_cat = nn.BatchNorm1d(params.X_dim + params.e_dim)
        # else:
        #     self.l0 = nn.Linear(params.X_dim + params.y_dim, params.H_dim, bias=True)
        #     self.bn_cat = nn.BatchNorm1d(params.X_dim + params.y_dim)

        self.bn_x = nn.BatchNorm1d(params.X_dim)
        if(self.params.layer_y):
            self.bn_y = nn.BatchNorm1d(params.e_dim)
            self.l0 = nn.Linear(params.X_dim + params.e_dim, params.H_dim, bias=True)
            self.l2 = nn.Linear(params.H_dim + params.e_dim, params.h_dim, bias=True)
        else:
            self.bn_y = nn.BatchNorm1d(params.y_dim)
            self.l0 = nn.Linear(params.X_dim + params.y_dim, params.H_dim, bias=True)
            self.l2 = nn.Linear(params.H_dim + params.y_dim, params.h_dim, bias=True)

        self.relu = nn.ReLU()
        self.drp_5 = nn.Dropout(.5)
        self.drp_1 = nn.Dropout(.1)
        self.bn = nn.BatchNorm1d(params.H_dim)
        self.bn_1 = nn.BatchNorm1d(params.h_dim)
        # ---------------------------------------------------------------
        self.mu = nn.Linear(params.h_dim, params.Z_dim, bias=True)
        self.var = nn.Linear(params.h_dim, params.Z_dim, bias=True)

        weights_init(self.l0.weight)
        weights_init(self.l2.weight)
        weights_init(self.mu.weight)
        weights_init(self.var.weight)

    def forward(self, X, y, eps=1e-7):

        if(self.params.layer_y):
            y_sum = torch.sum(y, dim=1).view(-1,1) + eps
            y = torch.mm(y, Variable(self.w2v_w))
            y = y/y_sum
        y = self.bn_y(y)
        o = self.bn_x(X)
        o = torch.cat((y,o), dim=-1)
        # ---------------------------------------
        o = self.l0(o)
        o = self.relu(o)
        # ---------------------------------------
        o = torch.cat((y,o), dim=-1)
        o = self.l2(o)
        o = self.relu(o)
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
        self.l3 = nn.Sigmoid()
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
        o = self.relu(o)
        # ---------------------------------------
        o = self.drp_1(o)
        o = self.l2(o)
        o = self.l3(o)
        if(np.isnan(o.data.cpu().numpy()).any()):
            import pdb
            pdb.set_trace
        return o

class decoder(torch.nn.Module):
    
    def __init__(self, params):
        
        super(decoder, self).__init__()
        self.params = params

        # self.emb_layer = nn.Linear(params.y_dim, params.e_dim, bias=False) 
        # self.emb_layer.weight = torch.nn.Parameter(torch.from_numpy(params.w2v_w).type(params.dtype))
        # self.emb_layer.weight.requires_grad = False
        self.w2v_w = torch.from_numpy(params.w2v_w).type(params.dtype)
        if(self.params.layer_y):
            self.bn_cat = nn.BatchNorm1d(params.Z_dim + params.e_dim)
            self.l0 = nn.Linear(params.Z_dim + params.e_dim, params.H_dim, bias=True)
        else:
            self.bn_cat = nn.BatchNorm1d(params.Z_dim + params.y_dim)
            self.l0 = nn.Linear(params.Z_dim + params.y_dim, params.H_dim, bias=True)
        self.bn_l0 = nn.BatchNorm1d(params.H_dim)
        # ==================================================
        self.l2 = nn.Linear(params.H_dim, params.X_dim, bias=True)
        self.relu = nn.ReLU()
        self.drp_5 = nn.Dropout(.5)
        self.drp_1 = nn.Dropout(.1)
        self.l3 = nn.Sigmoid()
        weights_init(self.l0.weight)
        weights_init(self.l2.weight)
        # ==================================================

    def forward(self, z, y, eps=1e-7):
        
        if(self.params.layer_y):
            y_sum = torch.sum(y, dim=1).view(-1,1) + eps
            y = torch.mm(y, Variable(self.w2v_w))
            o = y/y_sum
        else:
            o = y
        o = torch.cat((o, z), dim=-1)
        o = self.bn_cat(o)
        o = self.l0(o)
        o = self.bn_l0(o)
        o = self.relu(o)
        o = self.l2(o)
        o = self.l3(o)
        if(np.isnan(o.data.cpu().numpy()).any()):
            import pdb
            pdb.set_trace
        return o


class decoderRes(torch.nn.Module):
    
    def __init__(self, params):
        
        super(decoderRes, self).__init__()
        self.params = params

        # self.emb_layer = nn.Linear(params.y_dim, params.e_dim, bias=False) 
        # self.emb_layer.weight = torch.nn.Parameter(torch.from_numpy(params.w2v_w).type(params.dtype))
        # self.emb_layer.weight.requires_grad = False
        self.w2v_w = torch.from_numpy(params.w2v_w).type(params.dtype)
        self.bn_z = nn.BatchNorm1d(params.Z_dim)
        if(self.params.layer_y):
            self.bn_y = nn.BatchNorm1d(params.e_dim)
            self.l0 = nn.Linear(params.Z_dim + params.e_dim, params.H_dim, bias=True)
            self.l2 = nn.Linear(params.H_dim + params.e_dim, params.X_dim, bias=True)
        else:
            self.bn_y = nn.BatchNorm1d(params.y_dim)
            self.l0 = nn.Linear(params.Z_dim + params.y_dim, params.H_dim, bias=True)
            self.l2 = nn.Linear(params.H_dim + params.y_dim, params.X_dim, bias=True)
        self.bn_l0 = nn.BatchNorm1d(params.H_dim)
        self.relu = nn.ReLU()
        self.drp_5 = nn.Dropout(.5)
        self.drp_1 = nn.Dropout(.1)
        self.l3 = nn.Sigmoid()
        weights_init(self.l0.weight)
        weights_init(self.l2.weight)

    def forward(self, z, y, eps=1e-7):
        
        if(self.params.layer_y):
            y_sum = torch.sum(y, dim=1).view(-1,1) + eps
            y = torch.mm(y, Variable(self.w2v_w))
            y = y/y_sum
            y = self.bn_y(y)
        else:
            y = self.bn_y(y)
        o = self.bn_z(z)
        o = torch.cat((y, o), dim=-1)
        o = self.l0(o)
        o = self.bn_l0(o)
        o = self.relu(o)
        o = torch.cat((y, o), dim=-1)
        o = self.l2(o)
        o = self.l3(o)
        if(np.isnan(o.data.cpu().numpy()).any()):
            import pdb
            pdb.set_trace
        return o
