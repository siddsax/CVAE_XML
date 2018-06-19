import os
import sys
import torch
import timeit
import argparse
import numpy as np
sys.path.append('utils/')
sys.path.append('models/')
import data_helpers 
from perplexity import Perplexity
import time
from futils import *
import torch.nn as nn
from w2v import *
from visdom import Visdom
from embedding_layer import embedding_layer
from cnn_decoder import cnn_decoder
from cnn_encoder import cnn_encoder
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.autograd as autograd
from sklearn import preprocessing
from torch.autograd import Variable
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec
from scipy import sparse
import torch.cuda as cutorch

import subprocess

viz = Visdom()

if(torch.cuda.is_available()):
    use_cuda = 1
    dtype_f = torch.cuda.FloatTensor
    dtype_i = torch.cuda.LongTensor
    print("--------------- Using GPU! ---------")
else:
    use_cuda = 0
    dtype_f = torch.FloatTensor
    dtype_i = torch.LongTensor
    print("=============== Using CPU =========")

# -------------- Load data and params -------------------------------------------------
x_tr = np.load(sys.argv[1] + '/x_train.npy')
y_tr = sparse.load_npz(sys.argv[1] + '/y_train.npz').todense()
# x_te = np.load(sys.argv[1] + '/x_test.npy')
# y_te = sparse.load_npz(sys.argv[1] + '/y_test.npz').todense()
vocabulary = np.load(sys.argv[1] + '/vocab.npy').item()
vocabulary_inv = np.load(sys.argv[1] + '/vocab_inv.npy')
params = np.load(sys.argv[1] + '/params.npy').item()
params.classes = y_tr.shape[1]
params.decoder_kernels = [(400, params.Z_dim + params.classes + params.embedding_dim, 3),
                                (450, 400, 3),
                                (500, 450, 3)]
params.decoder_dilations = [1, 2, 4]
params.decoder_paddings = [effective_k(w, params.decoder_dilations[i]) - 1
                                 for i, (_, _, w) in enumerate(params.decoder_kernels)]
if(len(params.model_name)==0):
    params.model_name = gen_model_file(params)
if params.model_variation=='pretrain':
    embedding_weights = load_word2vec(params.pretrain_type, vocabulary_inv, params.embedding_dim)
else:
    embedding_weights = None
params.vocab_size = len(vocabulary)
go_row = np.ones((params.batch_size,1))*vocabulary[params.go_token]
end_row = np.ones((params.batch_size,1))*vocabulary[params.end_token]
# -------------------------------------------------------------------------------------

# -------------------------- Loss ---------------------------------------
# loss_fn = torch.nn.BCELoss(size_average=False)
loss_fn = torch.nn.BCELoss(size_average=False)
# loss_fn = torch.nn.MSELoss(size_average=False)
loss_best = float('Inf')
loss_best = float('Inf')
kl_b = float('Inf')
lk_b = float('Inf')
loss_best2 = float('Inf')
X_dim = x_tr.shape[1]
y_dim = y_tr.shape[1]
N_tr = x_tr.shape[0]
cnt = 0
#  --------------------------------------------------------------------

# ------------------------ GPU -------------------------------------------------
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor
# -------------------------------------------------------------------------------


# --------------------- GPU Config ----------------------------
a = 0
b = 1
emb = embedding_layer(params, embedding_weights)
enc = cnn_encoder(params)
dec = cnn_decoder(params)
if use_cuda:
    emb.cuda()
    enc.cuda(a)
    dec.cuda(b)
    print(emb);print(enc);print(dec);print("%"*100)
print("Number of Params : Embed {0}, Encoder {1}, Decoder {2}".format(count_parameters(emb), count_parameters(enc), count_parameters(dec)))
# ---------------------------------------------------------

# --------------- adam --------------------------------
optimizer = optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=params.lr)
# ------------------------------------------------------

if(params.training):
    for i in range(params.num_epochs):
        # ------------------ Load Batch Data ---------------------------------------------------------
        indexes = np.array(np.random.randint(N_tr, size=params.batch_size))
        batch_x = x_tr[indexes,:]
        batch_y = y_tr[indexes,:]
        decoder_word_input = np.concatenate((go_row,batch_x), axis=1)
        decoder_target = np.concatenate((batch_x, end_row), axis=1)

        batch_x = Variable(torch.from_numpy(batch_x.astype('int')).type(dtype_i))
        batch_y = Variable(torch.from_numpy(batch_y.astype('float')).type(dtype_f))
        decoder_word_input = Variable(torch.from_numpy(decoder_word_input.astype('int')).type(dtype_i))
        decoder_target = Variable(torch.from_numpy(decoder_target.astype('int')).type(dtype_i))
        decoder_target = decoder_target.view(-1)
        # -------------------------------------------------------------------------------------------

        # -------------- Encode ---------------------------------------------------
        e_emb = emb.forward(batch_x)
        z_mu, z_lvar = enc.forward(e_emb, batch_y)
        z = Variable(torch.randn([params.batch_size, params.Z_dim])).type(dtype_f)
        eps = torch.exp(0.5 * z_lvar)
        z = z * eps + z_mu
        kld = (-0.5 * torch.sum(z_lvar - torch.pow(z_mu, 2) - torch.exp(z_lvar) + 1, 1)).mean().squeeze()
        # -------------------------------------------------------------------------

        # --------------- Decoder --------------------------------------------------------------------
        decoder_input = emb.forward(decoder_word_input)
        logits = dec.forward(decoder_input.cuda(b), z.cuda(b), batch_y.cuda(b))
        logits = logits.cuda(a)
        logits = logits.view(-1, params.vocab_size)
        cross_entropy = torch.nn.functional.cross_entropy(logits, decoder_target)
        # --------------------------------------------------------------------------------------------        
        
        # --------------- Loss and Saving / Diagram Routines ------------------------------------------ 
        loss = params.beta*cross_entropy + kld
        print('Iter-{}; Loss: {:.4}; KL-loss: {:.4} ({}); recons_loss: {:.4} ({}); best_loss: {:.4};'.format(i, \
        loss.data, kld.data, kl_b, cross_entropy.data, lk_b, loss_best2))

        if(params.disp_flg):
            if(i==0):
                loss_old = loss.data
            else:
                viz.line(X=np.linspace(i - 1,i,50), Y=np.linspace(loss_old, loss.data,50), name='1', update='append', win=win)
                loss_old = loss.data
            if(i % 100 == 0 ):
                win = viz.line(X=np.arange(i, i + .1), Y=np.arange(0, .1))

        if(loss<loss_best2):
            loss_best2 = loss.data
            lk_b = cross_entropy.data
            kl_b = kld.data

        if params.save:
            
            if i % params.save_step == 0:
                if not os.path.exists('saved_models/' + params.model_name ):
                    os.makedirs('saved_models/' + params.model_name)
                torch.save(emb.state_dict(), "saved_models/" + params.model_name + "/emb_" + str(i))
                torch.save(enc.state_dict(), "saved_models/" + params.model_name + "/enc_"+ str(i))
                torch.save(dec.state_dict(), "saved_models/" + params.model_name + "/dec_"+ str(i))
                cnt += 1

            if(loss.data<loss_best):
                loss_best = loss.data
                torch.save(emb.state_dict(), "saved_models/" + params.model_name + "/emb_best")
                torch.save(enc.state_dict(), "saved_models/" + params.model_name + "/enc_best")
                torch.save(dec.state_dict(), "saved_models/" + params.model_name + "/dec_best")
        # -------------------------------------------------------------------------------------------------
        # -------------------------------- Loss Propogation and Optimization ------------------------------
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # -------------------------------------------------------------------------------------------------

# else:
#     seed = np.random.normal(size=[1, params.Z_dim])
#     seed = Variable(torch.from_numpy(seed).float().type(dtype_f))
#     if use_cuda:
#         seed = seed.cuda()

#     decoder_word_input_np, _ = batch_loader.go_input(1)
#     decoder_word_input = Variable(torch.from_numpy(decoder_word_input_np).long().type(dtype_i))

#     if use_cuda:
#         decoder_word_input = decoder_word_input.cuda()

#     result = ''

    # for i in range(seq_len):
    #     logits, _ = self(0., None, None,
    #                         decoder_word_input,
    #                         seed)

    #     [_, sl, _] = logits.size()

    #     logits = logits.view(-1, self.params.word_vocab_size)
    #     prediction = F.softmax(logits)
    #     prediction = prediction.view(1, sl, -1)

    #     # take the last word from prefiction and append it to result
    #     word = batch_loader.sample_word_from_distribution(prediction.data.cpu().numpy()[0, -1])

    #     if word == batch_loader.end_token:
    #         break

    #     result += ' ' + word

    #     word = np.array([[batch_loader.word_to_idx[word]]])

    #     decoder_word_input_np = np.append(decoder_word_input_np, word, 1)
    #     decoder_word_input = Variable(t.from_numpy(decoder_word_input_np).long())

    #     if use_cuda:
    #         decoder_word_input = decoder_word_input.cuda()

