from header import *

class variational(nn.Module):
    def __init__(self, params):
        super(variational, self).__init__()
        self.params = params
        self.l1 = nn.Linear(params.H_dim, params.h_dim)
        self.relu = nn.ReLU()
        self.bn_1 = nn.BatchNorm1d(params.h_dim)
        self.mu = nn.Linear(params.h_dim, params.Z_dim)
        self.var = nn.Linear(params.h_dim, params.Z_dim)
        if(self.params.dropouts):
            self.drp = nn.Dropout(p=.5)
        weights_init(self.l1.weight)
        weights_init(self.var.weight)
        weights_init(self.mu.weight)

    def forward(self, H):
        H = self.l1(H)
        H = self.relu(H)
        # H = self.bn_1(H)
        # if(self.params.dropouts):
        #     H = self.drp(H)
        return self.mu(H), self.var(H)