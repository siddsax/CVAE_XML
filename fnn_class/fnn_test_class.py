from header import *
from precision_k import precision_k
import time
import pdb
def model_test(model, X, Y):
    z_mu, z_var  = model.encoder.forward(X)
    # kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
    # ---------------------------------------------------------------
    
    # ----------- Decode (X, z) --------------------------------------------
    Y_sample = model.decoder.forward(z_mu)
    recon_loss = model.params.loss_fn(Y_sample, Y)
    loss = recon_loss + 0
    return Y_sample, loss, 0, recon_loss

def test(x_te, y_te, params, model=None, best_test_loss=None):

    if(model==None):
        model = fnn_model_class(params)
        model.load_state_dict(torch.load(params.load_model + "/model_best",  map_location=lambda storage, loc: storage))#, map_location=lambda storage, loc: storage)
    
    if(torch.cuda.is_available()):
        model = model.cuda()
        print("--------------- Using GPU! ---------")
    else:
        print("=============== Using CPU =========")

    model.eval()
    X, Y = load_data(x_te, y_te, params, batch=False)
    loss, recon_loss, lkhood_xy, kl_loss, Y_sample, X_sample = model(X, Y, test=1)

    if(best_test_loss!=None):
        if(loss.data[0] < best_test_loss ):
            best_test_loss = loss.data[0]
        print('Test Loss; Loss: {:.4}; KL-loss: {:.4}; recons_loss: {:.4}; best_loss: {:.4};'.format(loss.data[0], \
        -1.0, recon_loss, best_test_loss))
        p = precision_k(y_te, Y_sample, 5)
        # precision_k(y_te, Y_sample, 1)
        return best_test_loss, p
    else:
        print("==++==")
        # pdb.set_trace()
        print('Test Loss; Loss: {:.4}; KL-loss: {:.4}; recons_loss: {:.4}'.format(loss.data[0], kl_loss, recon_loss, \
        ))
        precision_k(y_te, Y_sample.data.numpy(), 5)
        np.save('scores', Y_sample.data[0])
        Y_probabs = sparse.csr_matrix(Y_sample.data[0])
        sio.savemat('score_matrix.mat' , {'score_matrix': Y_probabs})
