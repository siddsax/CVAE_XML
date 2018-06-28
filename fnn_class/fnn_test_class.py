from header import *

def model_test(model, X, Y):
    z_mu, z_var  = model.encoder.forward(X)
    kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
    # ---------------------------------------------------------------
    
    # ----------- Decode (X, z) --------------------------------------------
    Y_sample = model.decoder.forward(z_mu).data
    recon_loss = model.params.loss_fn(Y_sample, Y)
    loss = model.params.beta*recon_loss + kl_loss
    return Y_sample, loss, kl_loss, recon_loss

def test(x_te, y_te, params, model=None, best_test_loss=None):

    if(model==None):
        model = fnn_model_class(params)
        model.load_state_dict(torch.load("saved_models/" + params.model_name + "/model_best",  map_location=lambda storage, loc: storage))#, map_location=lambda storage, loc: storage)
    
    model.eval()

    X, Y = load_data(x_te, y_te, params, batch=False)
    Y_sample, loss, kl_loss, recon_loss = model_test(model, X, Y)

    if(best_test_loss!=None):
        if(loss < best_test_loss ):
            best_test_loss = loss
        print('Test Loss; Loss: {:.4}; KL-loss: {:.4}; recons_loss: {:.4}; best_loss: {:.4};'.format(loss, kl_loss, recon_loss, best_test_loss))
        return best_test_loss
    else:
        print('Test Loss; Loss: {:.4}; KL-loss: {:.4}; recons_loss: {:.4};'.format(loss.data, kl_loss.data, recon_loss.data))
        Y_probabs = sparse.csr_matrix(Y_sample.data)
        sio.savemat('score_matrix.mat' , {'score_matrix': Y_probabs})
