import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
import sys
sys.path.append('utils/')
sys.path.append('models/')
from sklearn import preprocessing
from sklearn.decomposition import PCA
import scipy.io as sio
from scipy import sparse
from encoder import encoder
from decoder import decoder
import argparse
from visdom import Visdom
from sklearn.externals import joblib 
from futils import *
from loss import loss
import time

viz = Visdom()
# ------------------------ Params -------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--pca', dest='pca_flag', type=int, default=0, help='1 to do pca, 0 for not doing it')
parser.add_argument('--zd', dest='Z_dim', type=int, default=200, help='Latent layer dimension')
parser.add_argument('--mb', dest='mb_size', type=int, default=100, help='Size of minibatch, changing might result in latent layer variance overflow')
parser.add_argument('--hd', dest='h_dim', type=int, default=600, help='hidden layer dimension')
parser.add_argument('--lr', dest='lr', type=int, default=1e-4, help='Learning Rate')
parser.add_argument('--p', dest='plot_flg', type=int, default=0, help='1 to plot, 0 to not plot')
parser.add_argument('--e', dest='num_epochs', type=int, default=100, help='step for displaying loss')
parser.add_argument('--b', dest='beta', type=float, default=1.0, help='factor multipied to likelihood param')
parser.add_argument('--d', dest='disp_flg', type=int, default=0, help='display graphs')
parser.add_argument('--sve', dest='save', type=int, default=1, help='save models or not')
parser.add_argument('--ss', dest='save_step', type=int, default=100, help='gap between model saves')
parser.add_argument('--mn', dest='model_name', type=str, default='', help='model name')
parser.add_argument('--tr', dest='training', type=int, default=1, help='model name')
parser.add_argument('--lm', dest='load_model', type=str, default="", help='model name')
parser.add_argument('--loss', dest='loss_type', type=str, default="L1Loss", help='model name')
parser.add_argument('--fl', dest='fin_layer', type=str, default="ReLU", help='model name')
parser.add_argument('--pp', dest='pp_flg', type=int, default=1, help='1 is for min-max pp, 2 is for gaussian pp, 0 for none')

# parser.add_argument('--ds', dest='data_set', type=str, default="Eurlex", help='dataset name')

args = parser.parse_args()
# --------------------------------------------------------------------------------------------------------------
if(len(args.model_name)==0):
    args.model_name = "Gen_data_Z_dim-{}_mb_size-{}_h_dim-{}_preproc-{}_beta-{}_final_ly-{}_loss-{}".format(args.Z_dim, args.mb_size, \
    args.h_dim, args.pp_flg, args.beta, args.fin_layer, args.loss_type)#, args.data_set)
print('Saving Model to: ' + args.model_name)
# ------------------ data ----------------------------------------------
print('Boom 0')

x_tr = np.load('datasets/Eurlex/eurlex_docs/x_tr.npy')
y_tr = np.load('datasets/Eurlex/eurlex_docs/y_tr.npy')

print('Boom 1')
# -------------------------- PP -------------------------------------------
if(args.pp_flg):
    if(args.pp_flg==1):
        pp = preprocessing.MinMaxScaler()
    elif(args.pp_flg==2):
        pp = preprocessing.StandardScaler()
    scaler = pp.fit(x_tr)
    x_tr = scaler.transform(x_tr)
print('Boom 2')
# ---------------------------------------------------------------------------

# -----------------------  Loss ------------------------------------
loss_fn = getattr(loss(), args.loss_type)
# -----------------------------------------------------------------
X_dim = x_tr.shape[1]
y_dim = y_tr.shape[1]
N = x_tr.shape[0]
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor
loss_best = 1e10
kl_b = 1e10
lk_b = 1e10
loss_best2 = 1e10
best_epch_loss = 1e10
best_test_loss = 1e10
num_mb = np.ceil(N/args.mb_size)
# ----------------------------------------------------------------

def sample_z(mu, log_var):
    eps = Variable(torch.randn(log_var.shape[0], args.Z_dim).type(dtype))
    k = torch.exp(log_var / 2) * eps
    return mu + k
# -----------------------------------------------------------------------------

if(args.training):
    if(len(args.load_model)):
        print(args.load_model)
        # P = nn.DataParallel(decoder(X_dim, y_dim, args.h_dim, args.Z_dim, args.fin_layer))
        # Q = nn.DataParallel(encoder(X_dim, y_dim, args.h_dim, args.Z_dim))
        # # original saved file with DataParallel
        # state_dictP = torch.load(args.load_model + "/P_best")
        # state_dictQ = torch.load(args.load_model + "/Q_best")
        # # create new OrderedDict that does not contain `module.`
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in state_dictP.items():
        #     name = k[7:] # remove `module.`
        #     new_state_dict[name] = v
        # # load params
        # P.load_state_dict(new_state_dict)
        # new_state_dict = OrderedDict()
        # for k, v in state_dictQ.items():
        #     name = k[7:] # remove `module.`
        #     new_state_dict[name] = v
        # # load params
        # Q.load_state_dict(new_state_dict)

        P = nn.DataParallel(torch.load(args.load_model + "/P_best"))
        Q = nn.DataParallel(torch.load(args.load_model + "/Q_best"))
    else:
        P = nn.DataParallel(decoder(X_dim, y_dim, args.h_dim, args.Z_dim, args.fin_layer))
        Q = nn.DataParallel(encoder(X_dim, y_dim, args.h_dim, args.Z_dim))
        if(torch.cuda.is_available()):
            P.cuda()
            Q.cuda()
            print("--------------- Using GPU! ---------")
        else:
            print("=============== Using CPU =========")

    optimizer = optim.Adam(list(P.parameters()) + list(Q.parameters()), lr=args.lr)

    print('Boom 5')
    # =============================== TRAINING ====================================
    for epoch in range(args.num_epochs):
        kl_epch = 0
        recon_epch = 0
        for it in range(int(num_mb)):
            # ---------------------------------- Sample Train Data -------------------------------
            a = np.random.randint(0,N, size=args.mb_size)#it*args.mb_size%(N-args.mb_size)
            X, Y = x_tr[a], y_tr[a]
            X = Variable(torch.from_numpy(X.astype('float32')).type(dtype))
            Y = Variable(torch.from_numpy(Y.astype('float32')).type(dtype))
            # -----------------------------------------------------------------------------------

            # ----------- Encode (X, Y) --------------------------------------------
            # inp = torch.cat([X, Y],1).type(dtype)
            z_mu, z_var  = Q.forward(X)
            z = sample_z(z_mu, z_var)
            kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
            # ---------------------------------------------------------------

            # ---------------------------- Decoder ------------------------------------
            inp = torch.cat([z, Y], 1).type(dtype)
            X_sample = P.forward(inp)
            recon_loss = loss_fn(X_sample, X)
            # -------------------------------------------------------------------------

            # ------------------ Check for Recon Loss ----------------------------
            if(recon_loss<0):
                print(recon_loss)
                print(X_sample[0:100])
                print(X[0:100])
                sys.exit()
            # ---------------------------------------------------------------------

            # ------------ Loss --------------------------------------------------
            loss = args.beta*recon_loss + kl_loss
            # --------------------------------------------------------------------

            #  --------------------- Print and plot  -------------------------------------------------------------------
            kl_epch += kl_loss.data
            recon_epch += recon_loss.data
            
            if it % int(num_mb/6) == 0:
                print('Loss: {:.4}; KL-loss: {:.4} ({}); recons_loss: {:.4} ({}); best_loss: {:.4};'.format(\
                loss.data, kl_loss.data, kl_b, recon_loss.data, lk_b, loss_best2))
                
                if(loss<loss_best2):
                    loss_best2 = loss
                    lk_b = recon_loss
                    kl_b = kl_loss

            # -------------------------------------------------------------------------------------------------------------- 
            
            # ------------------------ Propogate loss -----------------------------------
            loss.backward()
            del loss
            optimizer.step()
            optimizer.zero_grad()
            # ----------------------------------------------------------------------------
        
        kl_epch/= num_mb
        recon_epch/= num_mb
        loss_epch = kl_epch + args.beta*recon_epch
        if(loss_epch<best_epch_loss):
            best_epch_loss = loss_epch
            if not os.path.exists('saved_models/' + args.model_name ):
                os.makedirs('saved_models/' + args.model_name)
            torch.save(P, "saved_models/" + args.model_name + "/P_best")
            torch.save(Q, "saved_models/" + args.model_name + "/Q_best")

        print('End-of-Epoch: Epoch: {}; Loss: {:.4}; KL-loss: {:.4}; recons_loss: {:.4}; best_loss: {:.4};'.format(epoch, loss_epch, kl_epch, recon_epch, best_epch_loss))
        print("="*50)

        if args.save:
            if epoch % args.save_step == 0:
                if not os.path.exists('saved_models/' + args.model_name ):
                    os.makedirs('saved_models/' + args.model_name)
                torch.save(P, "saved_models/" + args.model_name + "/P_" + str(epoch))
                torch.save(Q, "saved_models/" + args.model_name + "/Q_"+ str(epoch))
        
        if(args.disp_flg):
            if(epoch==0):
                loss_old = loss_epch
            else:
                viz.line(X=np.linspace(epoch-1,epoch,50), Y=np.linspace(loss_old, loss_epch,50), name='1', update='append', win=win)
                loss_old = loss_epch
            if(epoch % 100 == 0 ):
                win = viz.line(X=np.arange(epoch, epoch + .1), Y=np.arange(0, .1))