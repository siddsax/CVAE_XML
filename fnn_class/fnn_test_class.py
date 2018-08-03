from header import *
from precision_k import precision_k
import time
import pdb

def test(x_te, y_te, params, model=None, best_test_loss=None):

    if(model==None):
        model = fnn_model_class(params)
        model = load_model(model, params.load_model + "/model_best_test")#, map_location=lambda storage, loc: storage)
        if(torch.cuda.is_available()):
            model = model.cuda()
            print("--------------- Using GPU! ---------")
        else:
            print("=============== Using CPU =========")


    model.eval()

    avg = 0.0
    candidates = [1, .1, .01, 0]
    can = candidates[np.random.randint(0,4)]
    print(can)
    X, Y = load_data(x_te, np.random.binomial(1, can, size=y_te.shape), params, batch=False)
    for i in range(5):
        loss, recon_loss, dist, dist_from_pred_y, kl_loss, Y_sample, X_sample, _, dist_l1, dist_bce, dist_mse = model(X, Y, test=1)
        avg+= dist

    X, Y = load_data(x_te, y_te, params, batch=False)
    zero_dist = model.params.loss_fns.logxy_loss(X, Variable(torch.zeros(X.shape).type(model.params.dtype)), model.params, f=model.params.loss_type).data.cpu().numpy()[0]
    loss, recon_loss, dist, dist_from_pred_y, kl_loss, Y_sample, X_sample, _, dist_l1, dist_bce, dist_mse = model(X, Y, test=1)
    model.train()

    if(best_test_loss!=None):
        if(loss.data[0] < best_test_loss ):
            best_test_loss = loss.data[0]
        print('Test Loss; Loss: {:.4}; KL-loss: {:.4}; recons_loss: {:.4}; best_loss: {:.4}; Rand {:.4}; All_min1_dist: {:.4};, dist: {:.4};, dist_from_pred_y: {:.4};, dist_l1 {:.4}, dist_bce {:.4}, dist_mse {:.4};'.format(loss.data[0], \
        kl_loss, recon_loss, best_test_loss, avg, zero_dist, dist, dist_from_pred_y, dist_l1, dist_bce, dist_mse))
        p = precision_k(y_te, Y_sample, 5)
        return best_test_loss, p, recon_loss, dist
    else:
        print("==++==")
        # pdb.set_trace()
        print('Test Loss; Loss: {:.4}; KL-loss: {:.4}; recons_loss: {:.4}; likelihood_x: {:.4}'.format(loss.data[0], kl_loss, recon_loss, \
         dist))
        precision_k(y_te, Y_sample, 5)
        np.save('scores', Y_sample)
        np.save('regen_data', X_sample)
        Y_probabs = sparse.csr_matrix(Y_sample)
        sio.savemat('score_matrix.mat' , {'score_matrix': Y_probabs})
