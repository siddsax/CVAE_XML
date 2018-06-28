from header import *
from fnn_train_gen import train
from dig_model import dig
sys.dont_write_bytecode = True
# from pycrayon import CrayonClient
# ------------------------ Params -------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--pca', dest='pca_flag', type=int, default=0, help='1 to do pca, 0 for not doing it')
parser.add_argument('--zd', dest='Z_dim', type=int, default=200, help='Latent layer dimension')
parser.add_argument('--mb', dest='mb_size', type=int, default=100, help='Size of minibatch, changing might result in latent layer variance overflow')
parser.add_argument('--hd', dest='h_dim', type=int, default=600, help='hidden layer dimension')
parser.add_argument('--lr', dest='lr', type=float, default=1e-4, help='Learning Rate')
parser.add_argument('--p', dest='plot_flg', type=int, default=0, help='1 to plot, 0 to not plot')
parser.add_argument('--e', dest='num_epochs', type=int, default=100, help='step for displaying loss')
parser.add_argument('--b', dest='beta', type=float, default=1.0, help='factor multipied to likelihood param')
parser.add_argument('--d', dest='disp_flg', type=int, default=0, help='display graphs')
parser.add_argument('--sve', dest='save', type=int, default=1, help='save models or not')
parser.add_argument('--ss', dest='save_step', type=int, default=100, help='gap between model saves')
parser.add_argument('--mn', dest='model_name', type=str, default='', help='model name')
parser.add_argument('--tr', dest='training', type=int, default=1, help='model name')
parser.add_argument('--te', dest='testing', type=int, default=0, help='model name')
parser.add_argument('--lm', dest='load_model', type=str, default="", help='model name')
parser.add_argument('--loss', dest='loss_type', type=str, default="L1Loss", help='model name')
parser.add_argument('--fl', dest='fin_layer', type=str, default="ReLU", help='model name')
parser.add_argument('--pp', dest='pp_flg', type=int, default=1, help='1 is for min-max pp, 2 is for gaussian pp, 0 for none')

# parser.add_argument('--ds', dest='data_set', type=str, default="Eurlex", help='dataset name')

params = parser.parse_args()
# --------------------------------------------------------------------------------------------------------------
if(len(params.model_name)==0):
    params.model_name = "Gen_data_Z_dim-{}_mb_size-{}_h_dim-{}_preproc-{}_beta-{}_final_ly-{}_loss-{}".format(params.Z_dim, params.mb_size, \
    params.h_dim, params.pp_flg, params.beta, params.fin_layer, params.loss_type)#, params.data_set)
print('Saving Model to: ' + params.model_name)
# ------------------ data ----------------------------------------------
print('Boom 0')
x_tr = np.load('../datasets/Eurlex/eurlex_docs/x_tr.npy')
y_tr = np.load('../datasets/Eurlex/eurlex_docs/y_tr.npy')

# -------------------------- PP -------------------------------------------
if(params.pp_flg):
    if(params.pp_flg==1):
        pp = preprocessing.MinMaxScaler()
    elif(params.pp_flg==2):
        pp = preprocessing.StandardScaler()
    scaler = pp.fit(x_tr)
    x_tr = scaler.transform(x_tr)
print('Boom 2')

# -----------------------  Loss ------------------------------------
params.loss_fn = getattr(loss(), params.loss_type)
# -----------------------------------------------------------------
params.X_dim = x_tr.shape[1]
params.y_dim = y_tr.shape[1]
params.N = x_tr.shape[0]
if torch.cuda.is_available():
    params.dtype = torch.cuda.FloatTensor
else:
    params.dtype = torch.FloatTensor

# ----------------------------------------------------------------

def sample_z(mu, log_var):
    eps = Variable(torch.randn(log_var.shape[0], params.Z_dim).type(dtype))
    k = torch.exp(log_var / 2) * eps
    return mu + k
# -----------------------------------------------------------------------------

if(params.training and not params.testing):
    train(x_tr, y_tr, params)
else:
    dig(x_tr, y_tr, params)
