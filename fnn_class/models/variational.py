# from header import *
# import pdb
# from weights_init import weights_init

# class encoder(torch.nn.Module):

#     def __init__(self, params):
        
#         super(encoder, self).__init__()
#         self.mu = nn.Linear(params.h_dim, params.Z_dim, bias=True)
#         self.var = nn.Linear(params.h_dim, params.Z_dim, bias=True)

#         weights_init(self.mu.weight)
#         weights_init(self.var.weight)
        
#     def forward(self, o):
        
#         o_ = self.mu(o)
#         o__ = self.var(o)
#         return o_, o__
