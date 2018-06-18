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

def get_gpu_memory_map(boom):
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ])
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    print("Print: {0}; Mem(1): {1}; Mem(2): {2}; Mem(3): {3}; Mem(4): {4}".format( boom, gpu_memory_map[0], \
    gpu_memory_map[1], gpu_memory_map[2], gpu_memory_map[3]))
    return boom+1
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

# loss_fn = torch.nn.BCELoss(size_average=False)
loss_fn = torch.nn.BCELoss(size_average=False)
# loss_fn = torch.nn.MSELoss(size_average=False)

# scaler = pp.fit(x_tr)
# x_tr = scaler.transform(x_tr)
# x_te = scaler.transform(x_te)

X_dim = x_tr.shape[1]
y_dim = y_tr.shape[1]
N_tr = x_tr.shape[0]
cnt = 0


if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

print 'model_variaton:', params.model_variation
if params.model_variation=='pretrain':
    embedding_weights = load_word2vec(params.pretrain_type, vocabulary_inv, params.embedding_dim)
else:
    embedding_weights = None

params.vocab_size = len(vocabulary)
boom = -3
emb = embedding_layer(params, embedding_weights)
enc = cnn_encoder(params)
dec = cnn_decoder(params)

if use_cuda:
    #boom = get_gpu_memory_map(boom) #1
    emb.cuda()
    #boom = get_gpu_memory_map(boom) #2
    enc.cuda()
    #boom = get_gpu_memory_map(boom) #3
    dec.cuda()
    #boom = get_gpu_memory_map(boom) #4

    print(emb)
    print(enc)
    print(dec)
    print("%"*100)
print("Number of Params : Embed {0}, Encoder {1}, Decoder {2}".format(count_parameters(emb), count_parameters(enc), count_parameters(dec)))

go_row = np.ones((params.batch_size,1))*vocabulary[params.go_token]
end_row = np.ones((params.batch_size,1))*vocabulary[params.end_token]

loss_fn = torch.nn.BCELoss(size_average=False)
perplexity = Perplexity()
optimizer = optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=params.lr)
cnt = 0

loss_best = float('Inf')
if(params.training):
    for i in range(params.num_epochs):
        boom = 1
        #boom = get_gpu_memory_map(boom) #1
        #boom = get_gpu_memory_map(boom) #2

        indexes = np.array(np.random.randint(N_tr, size=params.batch_size))
        batch_x = x_tr[indexes,:]
        batch_y = y_tr[indexes,:]
        decoder_word_input = np.concatenate((go_row,batch_x), axis=1)
        decoder_target = np.concatenate((batch_x, end_row), axis=1)

        #boom = get_gpu_memory_map(boom) #3
        batch_x = Variable(torch.from_numpy(batch_x.astype('int')).type(dtype_i))
        batch_y = Variable(torch.from_numpy(batch_y.astype('float')).type(dtype_f))
        decoder_word_input = Variable(torch.from_numpy(decoder_word_input.astype('int')).type(dtype_i))
        decoder_target = Variable(torch.from_numpy(decoder_target.astype('int')).type(dtype_i))
        #boom = get_gpu_memory_map(boom) #4

        #boom = get_gpu_memory_map(boom) #5
        e_emb = emb.forward(batch_x)
        #boom = get_gpu_memory_map(boom) #6
        z_mu, z_lvar = enc.forward(e_emb, batch_y)
        #boom = get_gpu_memory_map(boom) #7
        z = Variable(torch.randn([params.batch_size, params.Z_dim])).type(dtype_f)
        #boom = get_gpu_memory_map(boom) #8
        eps = torch.exp(0.5 * z_lvar)
        #boom = get_gpu_memory_map(boom) #9
        z = z * eps + z_mu
        #boom = get_gpu_memory_map(boom) #10
        decoder_input = emb.forward(decoder_word_input)
        #boom = get_gpu_memory_map(boom) #11
        logits = dec.forward(decoder_input, z, batch_y)
        #boom = get_gpu_memory_map(boom) #12
        kld = (-0.5 * torch.sum(z_lvar - torch.pow(z_mu, 2) - torch.exp(z_lvar) + 1, 1)).mean().squeeze()
        #boom = get_gpu_memory_map(boom) #13
        logits = logits.view(-1, params.vocab_size)
        #boom = get_gpu_memory_map(boom) #14
        
        decoder_target = decoder_target.view(-1)
        cross_entropy = torch.nn.functional.cross_entropy(logits, decoder_target)
        #boom = get_gpu_memory_map(boom) #15
        loss = params.beta*cross_entropy + kld
        #boom = get_gpu_memory_map(boom) #16
        kld = kld.data
        #boom = get_gpu_memory_map(boom) #17
        cross_entropy = cross_entropy.data
        #boom = get_gpu_memory_map(boom) #18
        loss.backward()
        #boom = get_gpu_memory_map(boom) #19
        optimizer.step()
        #boom = get_gpu_memory_map(boom) #20
        loss = loss.data
        #boom = get_gpu_memory_map(boom) #21
        optimizer.zero_grad()
        #boom = get_gpu_memory_map(boom) #22
        # ----------------------------------

        print("Loss = {0}, Iteration No. {1}, KL-Loss {2}, BCE-Loss {3}".format(loss, i, kld, cross_entropy))

        if(params.disp_flg):
            if(i==0):
                loss_old = loss
            else:
                viz.line(X=np.linspace(i - 1,i,50), Y=np.linspace(loss_old, loss,50), name='1', update='append', win=win)
                loss_old = loss
            if(i % 100 == 0 ):
                win = viz.line(X=np.arange(i, i + .1), Y=np.arange(0, .1))

        if params.save:
            
            if i % params.save_step == 0:
                if not os.path.exists('saved_models/' + params.model_name ):
                    os.makedirs('saved_models/' + params.model_name)
                torch.save(emb.state_dict(), "saved_models/" + params.model_name + "/emb_" + str(i))
                torch.save(enc.state_dict(), "saved_models/" + params.model_name + "/enc_"+ str(i))
                torch.save(dec.state_dict(), "saved_models/" + params.model_name + "/dec_"+ str(i))
                cnt += 1

            if(loss<loss_best):
                loss_best = loss
                torch.save(emb.state_dict(), "saved_models/" + params.model_name + "/emb_best")
                torch.save(enc.state_dict(), "saved_models/" + params.model_name + "/enc_best")
                torch.save(dec.state_dict(), "saved_models/" + params.model_name + "/dec_best")

else:
    seed = np.random.normal(size=[1, params.Z_dim])
    seed = Variable(torch.from_numpy(seed).float().type(dtype_f))
    if use_cuda:
        seed = seed.cuda()

    decoder_word_input_np, _ = batch_loader.go_input(1)
    decoder_word_input = Variable(torch.from_numpy(decoder_word_input_np).long().type(dtype_i))

    if use_cuda:
        decoder_word_input = decoder_word_input.cuda()

    result = ''

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

