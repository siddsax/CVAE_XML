from header import *
import pdb
from weights_init import weights_init

class encoder(torch.nn.Module):

    def __init__(self, params):
        
        super(encoder, self).__init__()

        # self.l0 = nn.Linear(params.X_dim, 1024, bias=True)
        # self.l5 = nn.Linear(1024, 512, bias=True)
        # self.l6 = nn.Linear(512, 256, bias=True)
        # self.l7 = nn.Linear(256, 128, bias=True)
        # self.l8 = nn.Linear(128, 64, bias=True)
        # self.mu = nn.Linear(params.h_dim, params.y_dim, bias=True)

        self.l0 = nn.Linear(params.X_dim, params.h_dim, bias=True)
        self.l1 = nn.ReLU()
        self.bn = nn.BatchNorm1d(params.h_dim)
        self.mu = nn.Linear(params.h_dim, params.Z_dim, bias=True)
        self.var = nn.Linear(params.h_dim, params.Z_dim, bias=True)
        
        weights_init(self.l0.weight)
        weights_init(self.mu.weight)
        weights_init(self.var.weight)
        # self.cc = CrayonClient(hostname="localhost")
        # self.summary = self.cc.create_experiment(type(self).__name__)

        
    def forward(self, inputs):
        
        o = self.l0(inputs)
        # if(torch.isnan(o).any()):
        #     print(torch.isnan(inputs).any())
        #     print(torch.isnan(o).any())
        #     print("Boobs")
        #     sys.exit()
        o = self.l1(o)
        # o = self.l5(o)
        # o = self.l1(o)
        # o = self.l6(o)
        # o = self.l1(o)
        # o = self.l7(o)
        # o = self.l1(o)
        # o = self.l8(o)
        # o = self.l1(o)
        # if(torch.isnan(o).any()):
        #     np.set_printoptions(threshold='nan')
        #     print(inputs)
        #     print(self.l0.weight)
        #     print(torch.isnan(self.l0.weight).any())
        #     print(torch.isnan(self.l0.bias).any())
        #     sys.exit()
        o = self.bn(o)
        o_ = self.mu(o)
        o__ = self.var(o)
        # if(torch.isnan(o__).any()):
        #     print(torch.isnan(inputs).any())
        #     print(torch.isnan(o_).any())
        #     print(torch.isnan(o__).any())
        #     print("Boobs")
        #     sys.exit()
        return o_, o__
