from header import *
import pdb
sys.dont_write_bytecode = True

# ------------------------ Params -------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--pca', dest='pca_flag', type=int, default=0, help='1 to do pca, 0 for not doing it')
parser.add_argument('--mb', dest='mb_size', type=int, default=100, help='Size of minibatch, changing might result in latent layer variance overflow')

parser.add_argument('--zd', dest='Z_dim', type=int, default=200, help='Latent layer dimension')
parser.add_argument('--hd', dest='h_dim', type=int, default=256, help='hidden layer dimension')
parser.add_argument('--Hd', dest='H_dim', type=int, default=512, help='hidden layer dimension')

parser.add_argument('--lr', dest='lr', type=float, default=1e-4, help='Learning Rate')
parser.add_argument('--p', dest='plot_flg', type=int, default=0, help='1 to plot, 0 to not plot')
parser.add_argument('--e', dest='num_epochs', type=int, default=10000, help='step for displaying loss')
parser.add_argument('--b', dest='beta', type=float, default=1.0, help='factor multipied to likelihood param')
parser.add_argument('--d', dest='disp_flg', type=int, default=0, help='display graphs')
parser.add_argument('--sve', dest='save', type=int, default=1, help='save models or not')
parser.add_argument('--sstep', dest='save_step', type=int, default=500, help='gap between model saves')
parser.add_argument('--mn', dest='model_name', type=str, default='', help='model name')
parser.add_argument('--tr', dest='training', type=int, default=1, help='model name')
parser.add_argument('--te', dest='testing', type=int, default=0, help='model name')
parser.add_argument('--lm', dest='load_model', type=str, default="", help='model name')
parser.add_argument('--ds', dest='data_set', type=str, default="Eurlex", help='dataset name')
parser.add_argument('--fl', dest='fin_layer', type=str, default="Sigmoid", help='model name')
parser.add_argument('--pp', dest='pp_flg', type=int, default=1, help='1 is for min-max pp, 2 is for gaussian pp, 0 for none')
parser.add_argument('--loss', dest='loss_type', type=int, default=2, help='model name MSE Default')
parser.add_argument('--clip', dest='clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--trlb', dest='train_labels', type=int, default=1, help='train on labeled data')
parser.add_argument('--ss', dest='ss', type=int, default=0, help='train on labeled data')
parser.add_argument('--ly', dest='layer_y', type=int, default=1, help='layer over labels')
parser.add_argument('--f', dest='freezing', type=int, default=0, help='layer over labels')

params = parser.parse_args()

# --------------------------------------------------------------------------------------------------------------
if(len(params.model_name)==0):
    params.model_name = "Classify_MLP_Z_dim-{}_mb_size-{}_h_dim-{}_pp_flg-{}_beta-{}_dataset-{}_final_ly-{}_loss-{}".format(params.Z_dim, params.mb_size, \
    params.h_dim, params.pp_flg, params.beta, params.data_set, params.fin_layer, params.loss_type)
print('Saving Model to: ' + params.model_name)
# ------------------ data ----------------------------------------------
print('Boom 0')

if(params.data_set=="Wiki"):
    x_tr = sparse.load_npz('/scratch/work/saxenas2/CVAE_XML/datasets/Wiki/x.npz')#.dense() # Prepocessed
    y_tr = sparse.load_npz('/scratch/work/saxenas2/CVAE_XML/datasets/Wiki/y.npz')#.dense()
    x_te = sparse.load_npz('/scratch/work/saxenas2/CVAE_XML/datasets/Wiki/tx.npz')#.dense()
    y_te = sparse.load_npz('/scratch/work/saxenas2/CVAE_XML/datasets/Wiki/ty.npz')#.dense()
elif(params.data_set=="Eurlex"):
    # x_tr = np.load('../datasets/Eurlex/manik/x_tr.npy')
    # y_tr = np.load('../datasets/Eurlex/manik/y_tr.npy')

    # x_tr = np.load('../datasets/Eurlex/manik/subsamples/x_20.npy')
    # y_tr = np.load('../datasets/Eurlex/manik/subsamples/y_20.npy')
    # x_te = np.load('../datasets/Eurlex/manik/x_te.npy')
    # y_te = np.load('../datasets/Eurlex/manik/y_te.npy')

    if(params.ss):
        x_unl = np.load('../datasets/Eurlex/eurlex_docs/x_tr.npy')
        params.ratio = 5
    else:
        x_unl = None
        params.ratio = 1

    x_tr = np.load('../datasets/Eurlex/eurlex_docs/x_20.npy')
    y_tr = np.load('../datasets/Eurlex/eurlex_docs/y_20.npy')
    x_for_pp = np.load('../datasets/Eurlex/eurlex_docs/x_tr.npy')
    x_te = np.load('../datasets/Eurlex/eurlex_docs/x_te.npy')
    y_te = np.load('../datasets/Eurlex/eurlex_docs/y_te.npy')

    params.w2v_w = np.load('../datasets/Eurlex/eurlex_docs/w2v_weights.npy')
    params.e_dim = params.w2v_w.shape[1]

# ----------------------------------------------------------------------
 
# x_tr = x_tr[0:20]
# y_tr = y_tr[0:20]

# labels = np.argwhere(np.sum(y_tr, axis=0)>0)
# lbl = [label[0] for label in labels]
# y_tr = y_tr[:,lbl]
# y_te = y_te[:,lbl]
# np.save('small_ytr', y_tr)

# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn import datasets, linear_model
# from sklearn.metrics import mean_squared_error, r2_score

# x = x_tr
# y = y_tr
# # pdb.set_trace()

# pred = np.zeros(np.shape(y))#np.load('scores.npy')
# np.set_printoptions(precision=4,threshold='nan')
# for i in range(y_tr.shape[1]):
#     regr = linear_model.LogisticRegression().fit(x, y[:,i])
#     p = regr.predict(x)
#     # print(p)
#     # print("-----")
#     # print(y[:,i])
#     # print("======================")
#     pred[:,i] = p

# # for i in range(y.shape[0]):
# #     a = np.argwhere(y[i,:]==1)
# #     a = [label[0] for label in a]
# #     print(pred[i,a])
# #     print("====")

# def L1Loss(X_sample, X):
#     t = np.mean(np.sum(np.abs(X_sample - X),axis=1))
#     return t
# # for i in range(y.shape[0]):
# #     print(pred[i])
# #     print("-----")
# #     print(y[i])
# #     a = np.argwhere(y[i]==1)
# #     a = [label[1] for label in a]
# #     print(pred[i,a])
# #     print("======================")
# print(L1Loss(pred, y))
# exit()


# -------------------------- PP -------------------------------------------
if(params.pp_flg):
    if(params.pp_flg==1):
        pp = preprocessing.MinMaxScaler()
    elif(params.pp_flg==2):
        pp = preprocessing.StandardScaler()
    
    scaler = pp.fit(x_for_pp)
    if(params.ss):
        x_unl = scaler.transform(x_unl)    

    x_tr = scaler.transform(x_tr)    
    x_te = scaler.transform(x_te)
    print('Boom 2')

# -----------------------  Loss ------------------------------------
params.loss_fns = loss()
# -----------------------------------------------------------------
params.X_dim = x_tr.shape[1]
params.y_dim = y_tr.shape[1]
params.N = x_tr.shape[0]
if (params.ss):
    params.N_unl = x_unl.shape[0]
else:
    params.N_unl = params.N
if torch.cuda.is_available():
    params.dtype = torch.cuda.FloatTensor
else:
    params.dtype = torch.FloatTensor

# -----------------------------------------------------------------------------

if(params.training and not params.testing):
    train(x_tr, y_tr, x_te, y_te, x_unl, params)
elif(params.testing):
    test(x_te, y_te, params)
else:
    dig(x_tr, y_tr, x_te, y_te, params)
