from header import *
def losses_update(list_of_losses, losses=None):
    if(losses is None):
        losses = np.zeros(len(list_of_losses))
    for i in range(len(list_of_losses)):
        losses[i] = list_of_losses[i]
    return losses

def losses_add(list_of_losses, losses=None):
    if(losses is None):
        losses = np.zeros(len(list_of_losses))
    for i in range(len(list_of_losses)):
        losses[i] += list_of_losses[i]
    return losses

def train(x_tr, y_tr, x_te, y_te, x_unl, params):
    viz = Visdom()
    loss_best = 1e10
    kl_b = 1e10
    lk_b = 1e10
    loss_best2 = 1e10
    num_mb = np.ceil(params.N/params.mb_size)
    best_epch_loss = 1e10
    best_test_loss = 1e10
    num_mb = np.ceil(params.N/params.mb_size)
    loss_names = ['loss', 'recon_loss', 'lkhood_xy', 'kl_loss', 'lossU', 'entropy', 'labeled_loss']
    model = fnn_model_class(params)
    if(torch.cuda.is_available()):
        model = model.cuda()
        print("--------------- Using GPU! ---------")
    else:
        print("=============== Using CPU =========")
    if(len(params.load_model)):
        print(params.load_model)
        model.load_state_dict(torch.load(params.load_model + "/model_best"))
        

    optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=params.lr)
    it2 = 0
    kt = 0
    p_best = np.zeros(5)
    for epoch in range(params.num_epochs):
        alpha = min(1.0, epoch*1e-3)#0.0
        kl_epch = 0
        recon_epch = 0
        for it in range(int(num_mb)):
            kt +=1
            X, Y = load_data(x_tr, y_tr, params)
            loss, recon_loss, lkhood_xy, kl_loss = model(X, Y)
            losses_new = [loss.data[0], recon_loss, lkhood_xy, kl_loss]

            if(x_unl is not None):
                # dummy = sparse.csr_matrix(np.zeros(np.shape(y_tr)))
                dummy = np.zeros(np.shape(x_unl))
                XU, _ = load_data(x_unl, dummy, params) ###### using dummy as dummy
                lossU, entropy, labeled_loss = model(XU)
                loss = loss + lossU
                lossU = lossU.data[0]
                losses_new.append(lossU.data[0], entropy, labeled_loss)

            losses = losses_add(losses_new, losses) if epoch or it else losses_add(losses_new)

            if it % max(int(num_mb/12),5) == 0:
                if(loss.data[0]<loss_best2):
                    loss_best2 = loss.data[0]
                    save_model(model,params, "/model_best_batch")
                out = ""
                for i in range(len(losses_new)):
                    out+= loss_names[i] + ":" + str(losses_new[i]) + " "
                print(out)
            # ------------------------ Propogate loss -----------------------------------
                # if(params.disp_flg):
                #     losses_now = [loss.data[0]]#, kl_loss, recon_loss]
                #     if(it2==0):
                #             losses = losses_update(losses_now)
                #     else:
                #             # print(losses)
                #             for j in range(len(losses)):
                #                 viz.line(X=np.linspace(it2-1,it2,50), Y=np.linspace(losses[j], losses_now[j],50),name=str(j), update='append', win=win)
                #             losses = losses_update(losses_now, losses)
                #     if(it2 % 100 == 0 ):
                #             win = viz.line(X=np.arange(it2, it2 + .1), Y=np.arange(0, .1))
                #     it2 += 1

            loss.backward()
            del loss
            torch.nn.utils.clip_grad_norm(model.parameters(), params.clip)
            # for p in model.parameters():
            #     p.data.add_(-params.lr, p.grad.data)
            optimizer.step()
            optimizer.zero_grad()
            # ----------------------------------------------------------------------------
        
        # ------------------- Save Model, Run on test data and find mean loss in epoch ----------------- 

        # for i in range(len(losses)):
        #     losses[i] /= num_mb
        # if(losses[0]<best_epch_loss):
        #     best_epch_loss = losses[0]
        #     save_model(model,params, "/model_best")
        # out=""
        # print("="*50)            
        # for i in range(len(losses_new)):
        #     out+= loss_names[i] + ":" + str(losses[i]) + " "
        # print(out)
        
        # best_test_loss,p_new = test(x_te, y_te, params, model=model, best_test_loss=best_test_loss)
        # model.train()
        
        # if(p_best[0]< p_new[0]):
        #     p_best = p_new
        #     print("====== New GOAT =====")
        #     save_model(model,params, "/model_best_test")
            
        # out = ""
        # for i in range(len(p_best)):
        #     out += str(i) + ":" + str(p_best[i]) + " "
        # print(out)
            
        # print("="*50)
