# from header import *
# import pdb
# from weights_init import weights_init

# class encoder(torch.nn.Module):

#     def __init__(self, params):
        
#         super(encoder, self).__init__()
#         self.l0 = nn.Linear(params.X_dim, params.h_dim, bias=True)
#         self.l1 = nn.ReLU()
#         self.drp = nn.Dropout(.5)
#         self.bn = nn.BatchNorm1d(params.h_dim)
#         self.mu = nn.Linear(params.h_dim, params.Z_dim, bias=True)
#         self.var = nn.Linear(params.h_dim, params.Z_dim, bias=True)

#         weights_init(self.l0.weight)
#         weights_init(self.mu.weight)
#         weights_init(self.var.weight)
        
#     def forward(self, inputs):
        
#         o = self.drp(inputs)
#         o = self.l0(o)
#         o = self.l1(o)
#         o = self.drp(o)
#         o = self.bn(o)
#         o_ = self.mu(o)
#         o__ = self.var(o)
#         return o_, o__
