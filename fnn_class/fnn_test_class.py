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



def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_vars(counter=0)
def test(x_te, y_te, params, model=None, best_test_loss=None):

    if(model==None):
        model = fnn_model_class(params)
        model = load_model(model, params.load_model + "/model_best_test")
        if(torch.cuda.is_available()):
            model = model.cuda()
            print("--------------- Using GPU! ---------")
        else:
            print("=============== Using CPU =========")
        
        # model = load_model(model, params.load_model + "/model_best_test", map_location=lambda storage, loc: storage)
    

    model.eval()
    
    avg = 0.0
    candidates = [1, .1, .01, 0]
    can = candidates[np.random.randint(0,4)]
    print(can)
    X, Y = load_data(x_te, np.random.binomial(1, can, size=y_te.shape), params, batch=False)
    for i in range(5):
        loss, recon_loss, dist, dist_from_pred_y, kl_loss, Y_sample, X_sample, X_from_pred_y = model(X, Y, test=1)
        avg+= dist

    test.counter +=1
    
    X, Y = load_data(x_te, y_te, params, batch=False)
    loss, recon_loss, dist, dist_from_pred_y, kl_loss, Y_sample, X_sample, X_from_pred_y = model(X, Y, test=1)
    model.train()

    if(best_test_loss!=None):
        if(loss.data[0] < best_test_loss ):
            best_test_loss = loss.data[0]
        print('Test Loss; Loss: {:.4}; KL-loss: {:.4}; recons_loss: {:.4}; best_loss: {:.4}; Rand_dist: {:.4};, dist: {:.4};, dist_from_pred_y: {:.4};'.format(loss.data[0], \
        kl_loss, recon_loss, best_test_loss, avg, dist, dist_from_pred_y))
        if(params.compress):
            p = np.zeros(5)
        else:
            p = precision_k(y_te, Y_sample, 5)
        return best_test_loss, p, recon_loss, dist
    else:
        print("==++==")
        # pdb.set_trace()
        print('Test Loss; Loss: {:.4}; KL-loss: {:.4}; recons_loss: {:.4}; likelihood_x: {:.4}'.format(loss.data[0], kl_loss, recon_loss, \
         dist))
        if(params.compress == 0):
            precision_k(y_te, Y_sample, 5)
            np.save('scores', Y_sample)
            Y_probabs = sparse.csr_matrix(Y_sample)
        
        np.save('regen_data', X_sample)
        sio.savemat('score_matrix.mat' , {'score_matrix': Y_probabs})
