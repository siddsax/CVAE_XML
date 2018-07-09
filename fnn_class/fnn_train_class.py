from header import *
def losses_update(list_of_losses, losses=None):
        if(losses is None):
                losses = np.zeros(len(list_of_losses))
        for i in range(len(list_of_losses)):
                losses[i] = list_of_losses[i]
        return losses

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
    else:
        
        if(torch.cuda.is_available()):
            model = model.cuda()
            print("--------------- Using GPU! ---------")
        else:
            print("=============== Using CPU =========")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=params.lr)
    it2 = 0
    kt = 0
    for epoch in range(params.num_epochs):
        alpha = min(1.0, epoch*10e3)#0.0
        kl_epch = 0
        recon_epch = 0
        for it in range(int(num_mb)):
            kt +=1
            X, Y = load_data(x_tr, y_tr, params)
            # loss, kl_loss, recon_loss = model(X, Y)
            loss = model(X, Y)
            # loss = alpha*kl_loss + params.beta*recon_loss
            # kl_loss = kl_loss.data
            recon_loss = loss.data
            # kl_epch += kl_loss
            recon_epch += recon_loss
            if it % int(num_mb/12) == 0:
                if(loss.data[0]<loss_best2):
                    loss_best2 = loss.data[0]
                    if not os.path.exists('saved_models/' + params.model_name ):
                        os.makedirs('saved_models/' + params.model_name)
                    torch.save(model.state_dict(), "saved_models/" + params.model_name + "/model_best")
                    print("-"*50)
                    # lk_b = recon_loss
                    # kl_b = kl_loss
                # print('Loss: {:.4}; KL-loss: {:.4} ({}); recons_loss: {:.4} ({}); best_loss: {:.4}'.format(\
                print('Loss: {:.4}; best_loss: {:.4}'.format(\
                loss.data[0], loss_best2))#kl_loss, kl_b, recon_loss, lk_b, loss_best2))
                # loss, kl_loss, kl_b, recon_loss, lk_b, loss_best2))
            # ------------------------ Propogate loss -----------------------------------
                if(params.disp_flg):
                    losses_now = [loss.data[0]]#, kl_loss, recon_loss]
                    if(it2==0):
                            losses = losses_update(losses_now)
                    else:
                            # print(losses)
                            for j in range(len(losses)):
                                viz.line(X=np.linspace(it2-1,it2,50), Y=np.linspace(losses[j], losses_now[j],50),name=str(j), update='append', win=win)
                            losses = losses_update(losses_now, losses)
                    if(it2 % 100 == 0 ):
                            win = viz.line(X=np.arange(it2, it2 + .1), Y=np.arange(0, .1))
                    it2 += 1
            
            
            
            loss.backward()
            del loss
            torch.nn.utils.clip_grad_norm(model.parameters(), params.clip)
            # for p in model.parameters():
            #     p.data.add_(-params.lr, p.grad.data)
            optimizer.step()
            # if(it % int(num_mb/3) == 0):
            #     thefile = open('gradient_classifier.txt', 'a+')
            #     write_grads(model, thefile)
            optimizer.zero_grad()
            # ----------------------------------------------------------------------------
        
        # ------------------- Save Model, Run on test data and find mean loss in epoch ----------------- 
        # print("="*50)            
        # kl_epch/= num_mb
        # recon_epch/= num_mb
        # loss_epch = recon_epch
        # if(loss_epch<best_epch_loss):
        #     best_epch_loss = loss_epch
        #     if not os.path.exists('saved_models/' + params.model_name ):
        #         os.makedirs('saved_models/' + params.model_name)
        #     torch.save(model.state_dict(), "saved_models/" + params.model_name + "/model_best")
        #     print("-"*50)
        # print('End-of-Epoch: Epoch: {}; Loss: {:.4}; KL-loss: {:.4}; recons_loss: {:.4}; best_loss: {:.4};'.\
        # format(epoch, loss_epch, kl_epch, recon_epch, best_epch_loss))
        # # best_test_loss = test(x_te, y_te, params, model=model, best_test_loss=best_test_loss)
        # print("="*50)
        
        # # --------------- Periodical Save and Display -----------------------------------------------------
        # if params.save and epoch % params.save_step == 0:
        #     if not os.path.exists('saved_models/' + params.model_name ):
        #         os.makedirs('saved_models/' + params.model_name)
        #     torch.save(model.state_dict(), "saved_models/" + params.model_name + "/model_"+ str(epoch))
