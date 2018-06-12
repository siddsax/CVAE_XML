import os
import sys
import torch
import timeit
import argparse
import numpy as np
sys.path.insert(0, 'utils')
sys.path.insert(0, 'models')
import data_helpers 
from perplexity import Perplexity

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



viz = Visdom()

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--pca', dest='pca_flag', type=int, default=0, help='1 to do pca, 0 for not doing it')
parser.add_argument('--zd', dest='Z_dim', type=int, default=500, help='Latent layer dimension')
parser.add_argument('--hd', dest='h_dim', type=int, default=750, help='hidden layer dimension')
parser.add_argument('--lr', dest='lr', type=int, default=1e-3, help='Learning Rate')
parser.add_argument('--p', dest='plot_flg', type=int, default=0, help='1 to plot, 0 to not plot')
parser.add_argument('--pp', dest='pp_flg', type=int, default=1, help='1 is for min-max pp, 0 is for gaussian pp')
parser.add_argument('--s', dest='step', type=int, default=100, help='step for displaying loss')
parser.add_argument('--b', dest='beta', type=int, default=1, help='Regularization param')
parser.add_argument('--d', dest='disp_flg', type=int, default=1, help='display graphs')
parser.add_argument('--sve', dest='save', type=int, default=0, help='save models or not')
parser.add_argument('--ss', dest='save_step', type=int, default=5000, help='gap between model saves')
parser.add_argument('--mn', dest='model_name', type=str, default='', help='model name')

parser.add_argument('--data_path',help='raw data path in CPickle format', type=str, default='datasets/rcv1_raw_small.p')
parser.add_argument('--sequence_length',help='max sequence length of a document', type=int,default=500)
parser.add_argument('--embedding_dim', help='dimension of word embedding representation', type=int, default=300)
parser.add_argument('--filter_sizes', help='number of filter sizes (could be a list of integer)', type=int, default=[2, 4, 8], nargs='+')
parser.add_argument('--num_filters', help='number of filters (i.e. kernels) in CNN model', type=int, default=32)
parser.add_argument('--pooling_units', help='number of pooling units in 1D pooling layer', type=int, default=32)
parser.add_argument('--pooling_type', help='max or average', type=str, default='max')
parser.add_argument('--hidden_dims', help='number of hidden units', type=int, default=512)
parser.add_argument('--model_variation', help='model variation: CNN-rand or CNN-pretrain', type=str, default='pretrain')
parser.add_argument('--pretrain_type', help='pretrain model: GoogleNews or glove', type=str, default='glove')
parser.add_argument('--batch_size', help='number of batch size', type=int, default=45)
parser.add_argument('--num_epochs', help='number of epcohs for training', type=int, default=50)
parser.add_argument('--vocab_size', help='size of vocabulary keeping the most frequent words', type=int, default=30000)
parser.add_argument('--training', help='training means 1, testing means 0', type=int, default=1)
parser.add_argument('--drop_prob', help='Dropout probability', type=int, default=.5)

params = parser.parse_args()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def effective_k(k, d):
    return (k - 1) * d + 1

def sample_z(mu, log_var):
    eps = Variable(torch.randn(params.batch_size, params.Z_dim).type(dtype_f))
    return mu + torch.exp(log_var / 2) * eps

def gen_model_file(params):
    data_name = params.data_path.split('/')[-2]
    fs_string = '-'.join([str(fs) for fs in params.filter_sizes])
    file_name = 'data-%s_sl-%d_ed-%d_fs-%s_nf-%d_pu-%d_pt-%s_hd-%d_bs-%d_model-%s_pretrain-%s' % \
        (data_name, params.sequence_length, params.embedding_dim,
         fs_string, params.num_filters, params.pooling_units,
         params.pooling_type, params.hidden_dims, params.batch_size,
         params.model_variation, params.pretrain_type)
    return file_name

def load_data(params):
    X_trn, Y_trn, X_tst, Y_tst, vocabulary, vocabulary_inv, params = data_helpers.load_data(params, max_length=params.sequence_length, vocab_size=params.vocab_size)
    X_trn = X_trn.astype(np.int32)
    X_tst = X_tst.astype(np.int32)
    Y_trn = Y_trn.astype(np.int32)
    Y_tst = Y_tst.astype(np.int32)
    return X_trn, Y_trn, X_tst, Y_tst, vocabulary, vocabulary_inv, params

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

params.decoder_kernels = [(400, params.Z_dim + params.embedding_dim, 3),
                                (450, 400, 3),
                                (500, 450, 3)]
params.decoder_dilations = [1, 2, 4]
params.decoder_paddings = [effective_k(w, params.decoder_dilations[i]) - 1
                                 for i, (_, _, w) in enumerate(params.decoder_kernels)]

if(len(params.model_name)==0):
    params.model_name = gen_model_file(params)



params.pad_token = "<PAD/>"
params.go_token = '<GO/>'
params.end_token = '<END/>'

print('-'*50)
print('Loading data...'); start_time = timeit.default_timer();
x_tr, y_tr, x_te, y_te, vocabulary, vocabulary_inv, params = load_data(params)
print('Process time %.3f (secs)\n' % (timeit.default_timer() - start_time))


# if(params.pp_flg):
#     pp = preprocessing.MinMaxScaler()
# else:
#     pp = preprocessing.StandardScaler()

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
emb = embedding_layer(params, embedding_weights)
enc = cnn_encoder(params)
dec = cnn_decoder(params)

if use_cuda:
    emb.cuda()
    enc.cuda()
    dec.cuda()

    print(emb)
    print(enc)
    print(dec)
    print("%"*100)
print("Number of Params : Embed {0}, Encoder {1}, Decoder {2}".format(count_parameters(emb), count_parameters(enc), count_parameters(dec)))

go_row = np.ones((params.batch_size,1))*vocabulary[params.go_token]
end_row = np.ones((params.batch_size,1))*vocabulary[params.end_token]

loss_fn = torch.nn.BCELoss(size_average=False)
perplexity = Perplexity()
optimizer = optim.Adam(list(emb.parameters()) + list(enc.parameters()) + list(dec.parameters()), lr=params.lr)

if(params.training):
    for i in range(params.num_epochs):

        indexes = np.array(np.random.randint(N_tr, size=params.batch_size))
        batch_x = x_tr[indexes,:]
        batch_y = y_tr[indexes,:]

        decoder_word_input = np.concatenate((go_row,batch_x), axis=1)
        decoder_target = np.concatenate((batch_x, end_row), axis=1)

        batch_x = Variable(torch.from_numpy(batch_x.astype('int')).type(dtype_i))
        decoder_word_input = Variable(torch.from_numpy(decoder_word_input.astype('int')).type(dtype_i))
        decoder_target = Variable(torch.from_numpy(decoder_target.astype('int')).type(dtype_i))

        
        e_emb = emb.forward(batch_x)
        z_mu, z_lvar = enc.forward(e_emb)
        z = sample_z(z_mu, z_lvar)
        decoder_input = emb.forward(decoder_word_input)
        logits = dec.forward(decoder_input, z)

        kld = (-0.5 * torch.sum(z_lvar - torch.pow(z_mu, 2) - torch.exp(z_lvar) + 1, 1)).mean().squeeze()

        logits = logits.view(-1, params.vocab_size)
        decoder_target = decoder_target.view(-1, 1)
        decoder_target_onehot = torch.FloatTensor(decoder_target.shape[0], params.vocab_size).type(dtype_f)
        decoder_target_onehot.zero_()
        decoder_target_onehot.scatter_(1, decoder_target, 1)
        cross_entropy = loss_fn(logits, decoder_target_onehot)

        # since cross enctropy is averaged over seq_len, it is necessary to approximate new kld
        loss = 79 * cross_entropy + kld

        logits = logits.view(params.batch_size, -1, params.vocab_size)
        decoder_target = decoder_target.view(params.batch_size, -1)
        ppl = perplexity(logits, decoder_target).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Loss = {0}, Iteration No. {1}".format(loss, i))

        # return ppl, kld

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

