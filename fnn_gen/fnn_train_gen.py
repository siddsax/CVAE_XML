from header import *

def train(x_tr, y_tr, params):
    viz = Visdom()
    kl_b = 1e10
    lk_b = 1e10
    loss_best2 = 1e10
    best_epch_loss = 1e10
    num_mb = np.ceil(params.N/params.mb_size)
    # thefile = open('gradient_classifier.txt', 'w')

    model = fnn_model_gen(params)
    if(len(params.load_model)):
        if(torch.cuda.is_available()):
            print("Loading from:" + params.load_model)
            model.load_state_dict(torch.load(params.load_model + "/model_best"))
            model = model.cuda()
            print("--------------- Using GPU! ---------")
        else:
            model.load_state_dict(torch.load(params.load_model + "/model_best", map_location=lambda storage, loc: storage))
            print("=============== Using CPU =========")


    optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=params.lr)

    for epoch in range(params.num_epochs):
        kl_epch = 0
        recon_epch = 0
        for it in range(int(num_mb)):
            X, Y = load_data(x_tr, y_tr, params)
            loss, kl_loss, recon_loss, x_pred = model(X, Y)

            #  --------------------- Print and plot  -------------------------------------------------------------------
            kl_epch += kl_loss.data
            recon_epch += recon_loss.data
            
            if it % int(num_mb/6) == 0:
                if(loss<loss_best2):
                    loss_best2 = loss
                    lk_b = recon_loss
                    kl_b = kl_loss
                    np.save('X_sample', x_pred.data.numpy())
                    np.save('X', X.numpy())

                print('Loss: {:.4}; KL-loss: {:.4} ({}); recons_loss: {:.4} ({}); best_loss: {:.4};'.format(\
                loss.data, kl_loss.data, kl_b, recon_loss.data, lk_b, loss_best2))
            # -------------------------------------------------------------------------------------------------------------- 
            
            # ------------------------ Propogate loss -----------------------------------
            loss.backward()
            del loss
            for p in model.parameters():
                if p.grad is not None:
                    p.data.add_(-params.lr, p.grad.data)
            optimizer.zero_grad()
            # ----------------------------------------------------------------------------
        
        kl_epch/= num_mb
        recon_epch/= num_mb
        loss_epch = kl_epch + params.beta*recon_epch
        if(loss_epch<best_epch_loss):
            best_epch_loss = loss_epch
            if not os.path.exists('saved_models/' + params.model_name ):
                os.makedirs('saved_models/' + params.model_name)
            torch.save(model.state_dict(), "saved_models/" + params.model_name + "/model_best")

        print('End-of-Epoch: Epoch: {}; Loss: {:.4}; KL-loss: {:.4}; recons_loss: {:.4}; best_loss: {:.4};'.format(epoch, loss_epch, kl_epch, recon_epch, best_epch_loss))
        print("="*50)

        # --------------- Periodical Save and Display -----------------------------------------------------
        if params.save and epoch % params.save_step == 0:
            if not os.path.exists('saved_models/' + params.model_name ):
                os.makedirs('saved_models/' + params.model_name)
            torch.save(model.state_dict(), "saved_models/" + params.model_name + "/model_"+ str(epoch))
        
        if(params.disp_flg):
            if(epoch==0):
                loss_old = loss_epch
            else:
                viz.line(X=np.linspace(epoch-1,epoch,50), Y=np.linspace(loss_old, loss_epch,50), name='1', update='append', win=win)
                loss_old = loss_epch
            if(epoch % 100 == 0 ):
                win = viz.line(X=np.arange(epoch, epoch + .1), Y=np.arange(0, .1))
        # --------------------------------------------------------------------------------------------------