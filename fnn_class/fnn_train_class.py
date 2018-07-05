from header import *

def train(x_tr, y_tr, x_te, y_te, params):
    viz = Visdom()
    loss_best = 1e10
    kl_b = 1e10
    lk_b = 1e10
    loss_best2 = 1e10
    num_mb = np.ceil(params.N/params.mb_size)
    best_epch_loss = 1e10
    best_test_loss = 1e10
    num_mb = np.ceil(params.N/params.mb_size)
    thefile = open('gradient_classifier.txt', 'w')

    model = fnn_model_class(params)
    if(len(params.load_model)):
        print(params.load_model)
        model.load_state_dict(torch.load(params.load_model + "/model_best"))
    if(torch.cuda.is_available()):
        model = model.cuda()
        print("--------------- Using GPU! ---------")
    else:
        print("=============== Using CPU =========")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=params.lr)

    for epoch in range(params.num_epochs):
        alpha = min(1.0, epoch*10e3)#0.0
        kl_epch = 0
        recon_epch = 0
        for it in range(int(num_mb)):
            X, Y = load_data(x_tr, y_tr, params)
            loss, kl_loss, recon_loss = model(X, Y)
            loss = alpha*kl_loss + params.beta*recon_loss
            kl_epch += kl_loss.data
            recon_epch += recon_loss.data
            if it % int(num_mb/3) == 0:
                if(loss<loss_best2):
                    loss_best2 = loss
                    lk_b = recon_loss
                    kl_b = kl_loss
                print('Loss: {:.4}; KL-loss: {:.4} ({}); recons_loss: {:.4} ({}); best_loss: {:.4}'.format(\
                loss, kl_loss, kl_b, recon_loss, lk_b, loss_best2))
            # ------------------------ Propogate loss -----------------------------------
            loss.backward()
            del loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), params.clip)
            for p in model.parameters():
                p.data.add_(-params.lr, p.grad.data)
            # optimizer.step()
            if(it % int(num_mb/3) == 0):
                thefile = open('gradient_classifier.txt', 'a+')
                write_grads(model, thefile)
            optimizer.zero_grad()
            # ----------------------------------------------------------------------------
        
        # ------------------- Save Model, Run on test data and find mean loss in epoch ----------------- 
        print("="*50)            
        kl_epch/= num_mb
        recon_epch/= num_mb
        loss_epch = kl_epch + params.beta*recon_epch
        if(loss_epch<best_epch_loss):
            best_epch_loss = loss_epch
            if not os.path.exists('saved_models/' + params.model_name ):
                os.makedirs('saved_models/' + params.model_name)
            torch.save(model.state_dict(), "saved_models/" + params.model_name + "/model_best")
        print('End-of-Epoch: Epoch: {}; Loss: {:.4}; KL-loss: {:.4}; recons_loss: {:.4}; best_loss: {:.4};'.\
        format(epoch, loss_epch, kl_epch, recon_epch, best_epch_loss))
        best_test_loss = test(x_te, y_te, params, model=model, best_test_loss=best_test_loss)
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
