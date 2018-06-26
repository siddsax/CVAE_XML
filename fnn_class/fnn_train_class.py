import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
#from tensorflow.examples.tutorials.mnist import input_data
import sys
sys.path.append('utils/')
sys.path.append('models/')
from sklearn import preprocessing
from sklearn.decomposition import PCA
import scipy.io as sio
from scipy import sparse
from encoder_classify import encoder
from decoder_classify import decoder
import argparse
from visdom import Visdom
from sklearn.externals import joblib 
from futils import *
from loss import loss
from fnn_model_class import fnn_model_class 
from fnn_test_class import test

def train(x_tr, y_tr, x_te, y_te, params):
    viz = Visdom()
    loss_best = 1e10
    kl_b = 1e10
    lk_b = 1e10
    loss_best2 = 1e10
    num_mb = np.ceil(params.N/params.mb_size)
    best_epch_loss = 1e10
    best_test_loss = 1e10
    num_mb = np.ceil(params.N/params.mb_size)
    thefile = open('gradient_classifier.txt', 'w')

    model = fnn_model_class(params)
    if(len(params.load_model)):
        print(params.load_model)
        model.load_state_dict(torch.load(params.load_model + "/model_best"))
    else:
        
        if(torch.cuda.is_available()):
            model = model.cuda()
            print("--------------- Using GPU! ---------")
        else:
            print("=============== Using CPU =========")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=params.lr)

    for epoch in range(params.num_epochs):
        kl_epch = 0
        recon_epch = 0
        for it in range(int(num_mb)):
            X, Y = load_data(x_tr, y_tr, params)
            loss, kl_loss, recon_loss = model(X, Y)
            kl_epch += kl_loss.data
            recon_epch += recon_loss.data
            if it % int(num_mb/3) == 0:
                if(loss<loss_best2):
                    loss_best2 = loss
                    lk_b = recon_loss
                    kl_b = kl_loss
                print('Loss: {:.4}; KL-loss: {:.4} ({}); recons_loss: {:.4} ({}); best_loss: {:.4}'.format(\
                loss, kl_loss, kl_b, recon_loss, lk_b, loss_best2))
            # ------------------------ Propogate loss -----------------------------------
            loss.backward()
            del loss
            optimizer.step()
            # ----------------------------------------------------------------------------
        
        # ------------------- Save Model, Run on test data and find mean loss in epoch ----------------- 
        print("="*50)            
        kl_epch/= num_mb
        recon_epch/= num_mb
        loss_epch = kl_epch + params.beta*recon_epch
        if(loss_epch<best_epch_loss):
            best_epch_loss = loss_epch
            if not os.path.exists('saved_models/' + params.model_name ):
                os.makedirs('saved_models/' + params.model_name)
            torch.save(model.state_dict(), "saved_models/" + params.model_name + "/model_best")
        print('End-of-Epoch: Epoch: {}; Loss: {:.4}; KL-loss: {:.4}; recons_loss: {:.4}; best_loss: {:.4};'.\
        format(epoch, loss_epch, kl_epch, recon_epch, best_epch_loss))
        thefile = open('gradient_classifier.txt', 'a+')
        write_grads(model, thefile)
        best_test_loss = test(x_te, y_te, params, model=model, best_test_loss=best_test_loss)
        optimizer.zero_grad()
        print("="*50)
        
        # --------------- Periodical Save and Display -----------------------------------------------------
        if params.save and epoch % params.save_step == 0:
            if not os.path.exists('saved_models/' + params.model_name ):
                os.makedirs('saved_models/' + params.model_name)
            torch.save(model.state_dict(), "saved_models/" + params.model_name + "/model_"+ str(epoch))
        
        if(params.disp_flg):
            if(epoch==0):
                loss_old = loss_epch
            else:
                viz.line(X=np.linspace(epoch-1,epoch,50), Y=np.linspace(loss_old, loss_epch,50), name='1', update='append', win=win)
                loss_old = loss_epch
            if(epoch % 100 == 0 ):
                win = viz.line(X=np.arange(epoch, epoch + .1), Y=np.arange(0, .1))
        # --------------------------------------------------------------------------------------------------