from header import *

class encoder(torch.nn.Module):

    def __init__(self, params):
        
        super(encoder, self).__init__()
        
        self.l0 = nn.Linear(params.X_dim, params.h_dim, bias=True)
        self.l1 = nn.ReLU()
        self.bn = nn.BatchNorm1d(params.h_dim)
        self.mu = nn.Linear(params.h_dim, params.Z_dim, bias=True)
        self.var = nn.Linear(params.h_dim, params.Z_dim, bias=True)
        
        torch.nn.init.xavier_uniform_(self.l0.weight)
        torch.nn.init.xavier_uniform_(self.mu.weight)
        torch.nn.init.xavier_uniform_(self.var.weight)
        # self.cc = CrayonClient(hostname="localhost")
        # self.summary = self.cc.create_experiment(type(self).__name__)

        
    def forward(self, inputs):
        
        o = self.l0(inputs)
        if(torch.isnan(o).any()):
            np.set_printoptions(threshold='nan')
            print(inputs)
            print(self.l0.weight)
            print(torch.isnan(self.l0.weight).any())
            print(torch.isnan(self.l0.bias).any())
            sys.exit()
        o = self.l1(o)
        o = self.bn(o)
        o_ = self.mu(o)
        o__ = self.var(o)
        if(torch.isnan(o__).any()):
            print(torch.isnan(inputs).any())
            print(torch.isnan(o_).any())
            print(torch.isnan(o__).any())
            print("Boobs")
            sys.exit()
        return o_, o__
