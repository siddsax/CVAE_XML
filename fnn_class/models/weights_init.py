import torch

def weights_init(m):
    if(torch.__version__=='0.4.0'):
    	torch.nn.init.xavier_uniform_(m)
    else:
	torch.nn.init.xavier_uniform(m)