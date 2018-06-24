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
import numpy as np
sys.path.append('utils/')
sys.path.append('models/')
import data_helpers 

from w2v import *
from embedding_layer import embedding_layer
from cnn_decoder import cnn_decoder
from cnn_encoder import cnn_encoder
from sklearn import preprocessing
from sklearn.decomposition import PCA
import scipy.io as sio
from scipy import sparse
import argparse
from visdom import Visdom
from sklearn.externals import joblib 
from futils import *
viz = Visdom()
# ------------------------ Params -------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--pca', dest='pca_flag', type=int, default=0, help='1 to do pca, 0 for not doing it')
parser.add_argument('--zd', dest='Z_dim', type=int, default=200, help='Latent layer dimension')
parser.add_argument('--mb', dest='mb_size', type=int, default=100, help='Size of minibatch, changing might result in latent layer variance overflow')
parser.add_argument('--hd', dest='h_dim', type=int, default=600, help='hidden layer dimension')
parser.add_argument('--lr', dest='lr', type=int, default=1e-3, help='Learning Rate')
parser.add_argument('--p', dest='plot_flg', type=int, default=0, help='1 to plot, 0 to not plot')
parser.add_argument('--e', dest='num_epochs', type=int, default=100, help='step for displaying loss')
parser.add_argument('--b', dest='beta', type=float, default=1, help='factor multipied to likelihood param')
parser.add_argument('--d', dest='disp_flg', type=int, default=0, help='display graphs')
parser.add_argument('--sve', dest='save', type=int, default=1, help='save models or not')
parser.add_argument('--ss', dest='save_step', type=int, default=5000, help='gap between model saves')
parser.add_argument('--mn', dest='model_name', type=str, default='', help='model name')
parser.add_argument('--tr', dest='training', type=int, default=1, help='model name')
parser.add_argument('--lm', dest='load_model', type=str, default="", help='model name')
parser.add_argument('--ds', dest='data_set', type=str, default="rcv", help='dataset name')
parser.add_argument('--fl', dest='fin_layer', type=str, default="ReLU", help='model name')
parser.add_argument('--pp', dest='pp_flg', type=int, default=1, help='1 is for min-max pp, 2 is for gaussian pp, 0 for none')

parser.add_argument('--sequence_length',help='max sequence length of a document', type=int,default=500)
parser.add_argument('--embedding_dim', help='dimension of word embedding representation', type=int, default=300)
parser.add_argument('--model_variation', help='model variation: CNN-rand or CNN-pretrain', type=str, default='pretrain')
parser.add_argument('--pretrain_type', help='pretrain model: GoogleNews or glove', type=str, default='glove')
parser.add_argument('--vocab_size', help='size of vocabulary keeping the most frequent words', type=int, default=30000)
parser.add_argument('--drop_prob', help='Dropout probability', type=int, default=.3)
parser.add_argument('--load_data', help='Load Data or not', type=int, default=1)

params = parser.parse_args()
params.pad_token = "<PAD/>"
params.go_token = '<GO/>'
params.end_token = '<END/>'
data_path = 'datasets/rcv/' + params.data_set

if(len(params.model_name)==0):
    params.model_name = "Gen_data_CNN_Z_dim-{}_mb_size-{}_h_dim-{}_preproc-{}_beta-{}_final_ly-{}_loss-{}".format(params.Z_dim, params.mb_size, \
    params.h_dim, params.pp_flg, params.beta, params.fin_layer, params.loss_type)#, params.data_set)
print('Saving Model to: ' + params.model_name)
# ------------------ data ----------------------------------------------
print('Boom 0')

if(params.load_data):
    print("Loading Data")
    print(data_path)
    x_tr, y_tr, x_te, y_te, vocabulary, vocabulary_inv, params = load_data(params)

    np.save(data_path + '/x_train', x_tr)
    sparse.save_npz(data_path + '/y_train', y_tr)
    sparse.save_npz(data_path + '/y_test', y_te)
    np.save(data_path + '/x_test', x_te)
    np.save(data_path + '/vocab', vocabulary)
    np.save(data_path + '/vocab_inv', vocabulary_inv)
x_tr = np.load(data_path + '/x_train.npy')
y_tr = sparse.load_npz(data_path + '/y_train.npz').todense()
# x_te = np.load(data_path + '/x_test.npy')
# y_te = sparse.load_npz(data_path + '/y_test.npz').todense()
vocabulary = np.load(data_path + '/vocab.npy').item()
vocabulary_inv = np.load(data_path + '/vocab_inv.npy')
# -------------------------------------------------------------------------------------

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

# -----------------------  Loss ------------------------------------
loss_fn = getattr(loss(), params.loss_type)
# -----------------------------------------------------------------

# -------------------------- Params ---------------------------------------------
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
go_row = np.ones((params.mb_size,1))*vocabulary[params.go_token]
end_row = np.ones((params.mb_size,1))*vocabulary[params.end_token]
X_dim = x_tr.shape[1]
y_dim = y_tr.shape[1]
N = x_tr.shape[0]
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor
loss_best = float('Inf')
kl_b = float('Inf')
lk_b = float('Inf')
loss_best2 = float('Inf')
best_epch_loss = float('Inf')
best_test_loss = float('Inf')
num_mb = np.ceil(N/args.mb_size)
# ---------------------------------------------------------------------------------


# -----------------------------------------------------------------------------
if(params.training):
    if(len(params.load_model)):
        print('loading saved_models/Z_dim-50_mb_size-100_h_dim-100_pp_flg-1_beta-1000_dataset-Eurlex/')
        emb = torch.load("saved_models/Z_dim-50_mb_size-100_h_dim-100_pp_flg-1_beta-1000_dataset-Eurlex/emb_best")
        enc = torch.load("saved_models/Z_dim-50_mb_size-100_h_dim-100_pp_flg-1_beta-1000_dataset-Eurlex/enc_best")
        dec = torch.load("saved_models/Z_dim-50_mb_size-100_h_dim-100_pp_flg-1_beta-1000_dataset-Eurlex/dec_best")
        print(emb);print(enc);print(dec);print("%"*100)
    else:
        emb = embedding_layer(params, embedding_weights)
        enc = cnn_encoder(params)
        dec = cnn_decoder(params
        a = 0
        b = 1
        if(torch.cuda.is_available()):
            emb.cuda()
            enc.cuda(a)
            dec.cuda(b)
            print("--------------- Using GPU! ---------")
        else:
            print("=============== Using CPU =========")
    print(emb);print(enc);print(dec);print("%"*100)
    print("Number of Params : Embed {0}, Encoder {1}, Decoder {2}".format(count_parameters(emb), count_parameters(enc), count_parameters(dec)))

    optimizer = optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=params.lr)

    print('Boom 5')
    # =============================== TRAINING ====================================
    for epoch in range(params.num_epochs):
        kl_epch = 0
        recon_epch = 0
        for i in range(int(num_mb)):
            # ------------------ Load Batch Data ---------------------------------------------------------
            indexes = np.array(np.random.randint(N, size=params.mb_size))
            batch_x, batch_y = x_tr[indexes,:], y_tr[indexes,:]
            decoder_word_input = np.concatenate((go_row,batch_x), axis=1)
            decoder_target = np.concatenate((batch_x, end_row), axis=1)
            batch_x = Variable(torch.from_numpy(batch_x.astype('int')).type(dtype_i))
            batch_y = Variable(torch.from_numpy(batch_y.astype('float')).type(dtype_f))
            decoder_word_input = Variable(torch.from_numpy(decoder_word_input.astype('int')).type(dtype_i))
            decoder_target = Variable(torch.from_numpy(decoder_target.astype('int')).type(dtype_i))
            decoder_target = decoder_target.view(-1)
            # -----------------------------------------------------------------------------------

            # ----------- Encode (X, Y) --------------------------------------------
            e_emb = emb.forward(batch_x)
            z_mu, z_lvar = enc.forward(e_emb, batch_y)
            z = Variable(torch.randn([params.mb_size, params.Z_dim])).type(dtype_f)
            eps = torch.exp(0.5 * z_lvar)
            z = z * eps + z_mu
            kl_loss = (-0.5 * torch.sum(z_lvar - torch.pow(z_mu, 2) - torch.exp(z_lvar) + 1, 1)).mean().squeeze()
            # -------------------------------------------------------------------------

            # ---------------------------- Decoder ------------------------------------
            decoder_input = emb.forward(decoder_word_input)
            logits = dec.forward(decoder_input.cuda(b), z.cuda(b), batch_y.cuda(b))
            logits = logits.cuda(a)
            logits = logits.view(-1, params.vocab_size)
            # recon_loss = loss_fn(X_sample, X)
            # Here the loss is between one-hot vectors and the final fc layer
            cross_entropy = torch.nn.functional.cross_entropy(logits, decoder_target)
            # ------------------ Check for Recon Loss ----------------------------
            if(cross_entropy<0):
                print(cross_entropy)
                print(X_sample[0:100])
                print(X[0:100])
                sys.exit()
            # ---------------------------------------------------------------------
          
            # ------------ Loss --------------------------------------------------
            loss = params.beta*cross_entropy + kl_loss
            # --------------------------------------------------------------------

            #  --------------------- Print and plot  -------------------------------------------------------------------
            kl_epch += kl_loss.data
            recon_epch += cross_entropy.data
            
            if i % int(num_mb/6) == 0:
                print('Iter-{}; Loss: {:.4}; KL-loss: {:.4} ({}); recons_loss: {:.4} ({}); best_loss: {:.4};'.format(i, \
                loss.data, kl_loss.data, kl_b, cross_entropy.data, lk_b, loss_best2))

                if(loss<loss_best2):
                    loss_best2 = loss.data
                    lk_b = cross_entropy.data
                    kl_b = kl_loss.data
                    
            # -------------------------------------------------------------------------------------------------------------- 
            
            # ------------------------ Propogate loss -----------------------------------
            loss.backward()
            del loss
            optimizer.step()
            optimizer.zero_grad()
            # ----------------------------------------------------------------------------

        kl_epch/= num_mb
        recon_epch/= num_mb
        loss_epch = kl_epch + recon_epch
        if(loss_epch<best_epch_loss):
            best_epch_loss = loss_epch
            if not os.path.exists('saved_models/' + params.model_name ):
                os.makedirs('saved_models/' + params.model_name)
            torch.save(emb, "saved_models/" + params.model_name + "/emb_best")
            torch.save(enc, "saved_models/" + params.model_name + "/enc_best")
            torch.save(dec, "saved_models/" + params.model_name + "/dec_best")

        print('End-of-Epoch: Loss: {:.4}; KL-loss: {:.4}; recons_loss: {:.4}; best_loss: {:.4};'.format(loss_epch, kl_epch, recon_epch, best_epch_loss))
        print("="*50)

        if params.save:
            if epoch % params.save_step == 0:
                torch.save(emb, "saved_models/" + params.model_name + "/emb_best")
                torch.save(enc, "saved_models/" + params.model_name + "/enc_best")
                torch.save(dec, "saved_models/" + params.model_name + "/dec_best")

        if(params.disp_flg):
            if(epoch==0):
                loss_old = loss_epch
            else:
                viz.line(X=np.linspace(epoch-1,epoch,50), Y=np.linspace(loss_old, loss_epch,50), name='1', update='append', win=win)
                loss_old = loss_epch
            if(epoch % 100 == 0 ):
                win = viz.line(X=np.arange(epoch, epoch + .1), Y=np.arange(0, .1))
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

