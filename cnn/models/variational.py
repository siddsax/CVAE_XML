from header import *

class variational(nn.Module):
    def __init__(self, params):
        super(variational, self).__init__()
        self.params = params
        self.l1 = nn.Linear(params.H_dim, params.h_dim)
        self.relu = nn.ReLU()
        self.mu = nn.Linear(params.h_dim, params.Z_dim)
        self.var = nn.Linear(params.h_dim, params.Z_dim)

    def forward(self, H):
        H = self.l1(H)
        H = self.relu(H)
        return self.mu(H), self.var(H)