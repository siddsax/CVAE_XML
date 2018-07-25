from header import *
import pdb

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
    num_mb = np.ceil(params.N/(params.mb_size/params.ratio))
    best_epch_loss = 1e10
    best_test_loss = 1e10
    init = 0
    loss_names = ['lossF', 'lossL', 'recon_loss', 'dist', 'kl_loss', 'lossU', 'entropy', 'labeled_loss']
    model = fnn_model_class(params)
    
    if not os.path.exists('saved_models/' + params.model_name ):
        os.makedirs('saved_models/' + params.model_name)
    logs = open("saved_models/" + params.model_name + "/logs.txt", 'w+')
    # pdb.set_trace()
    logs.write(str(model))
    logs.write('\n')

    optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=params.lr)
    if(torch.cuda.is_available()):
        model = model.cuda()
        print("--------------- Using GPU! ---------")
    else:
        print("=============== Using CPU =========")
    if(len(params.load_model)):
        print(params.load_model)
        model, optimizer, init = load_model(model, params.load_model + "/model_best_test", optimizer)

    prem = 0
    kt = 0
    recon_loss_tst_old = 0
    re_tr_old = 0.0
    xlk_tr_old = 0.0
    tr_old = 0.0
    done = 0
    xlk_tr_tst_bst = 1e10

    p_best = np.zeros(5)
    for epoch in range(init, params.num_epochs):
        logs = open("saved_models/" + params.model_name + "/logs.txt", 'a+')
        for it in range(int(num_mb)):
            kt +=1
            if(params.train_labels):
                params.mb_size /= params.ratio
                X, Y = load_data(x_tr, y_tr, params)
                params.mb_size *= params.ratio 
                lossL, recon_loss, dist, kl_loss = model(X, Y)
                losses_new = [lossL.data[0], recon_loss, dist, kl_loss]
            else:
                losses_new = [0.0, 0.0, 0.0, 0.0]

            if(x_unl is not None):
                dummy = np.zeros(np.shape(x_unl))
                XU, _ = load_data(x_unl, dummy, params) ###### using dummy as dummy
                lossU, entropy, labeled_loss = model(XU)
                losses_new += [lossU.data[0], entropy, labeled_loss]

            if(x_unl is not None and params.train_labels): 
                lossF = lossL + lossU
            elif(params.train_labels):
                lossF = lossL
            elif(x_unl is not None):
                lossF = lossU
            else:
                print("Error, neither labeled or unlabeled data give")
                exit()                    
            losses_new = [lossF.data[0]] + losses_new

            # ------------------------ Propogate loss -----------------------------------
            lossF.backward()
            optimizer.step()
            optimizer.zero_grad()

            # ----------------------------------------------------------------------------
            losses = losses_add(losses_new, losses) if prem else losses_add(losses_new)
            if it % max(int(num_mb/12),5) == 0:
                out = ""
                for i in range(len(losses_new)):
                    out+= loss_names[i] + ":" + str(losses_new[i]) + " "
                
                zero_dist = model.params.loss_fns.logxy_loss(X, Variable(torch.zeros(X.shape).type(model.params.dtype)), model.params).data.cpu().numpy()[0]
                out += "zero loss" + ":" + str(zero_dist)
                print(out)
            prem = 1        
        # ------------------- Save Model, Run on test data and find mean loss in epoch ----------------- 
        for i in range(len(losses)):
            losses[i] /= num_mb
        if(losses[0]<best_epch_loss):
            best_epch_loss = losses[0]
            save_model(model, optimizer, epoch, params, "/model_best")
        out="Model Name :" + params.model_name + " Epoch No: " + str(epoch) + " "
        print("="*50)            
        for i in range(len(losses_new)):
            out+= loss_names[i] + ":" + str(losses[i]) + " "
        print(out)
        
        best_test_loss,p_new, recon_loss_tst, xlk_tr_tst = test(x_te, y_te, params, model=model, best_test_loss=best_test_loss)
        out = out + "recon_loss_tst: " + str(recon_loss_tst) + " P_1 " + str(p_new[0]) + "\n"
        logs.write(out)
        logs.close()
        
        if(epoch%100==0):
            plt.gcf().clear()
        if(done==1):
            ax1 = plt.subplot(3, 1, 1)
            ax1.scatter(np.linspace(epoch-1,epoch,50), np.linspace(recon_loss_tst_old, recon_loss_tst,50),color='blue', s=1)
            ax1.scatter(np.linspace(epoch-1,epoch,50), np.linspace(re_tr_old, losses[2],50),color='red', s=1)
            ax1.set_ylim(bottom=0)
            
            ax2 = plt.subplot(3, 1, 2)
            ax2.scatter(np.linspace(epoch-1,epoch,50), np.linspace(xlk_tr_old_tst, xlk_tr_tst,50),color='blue', s=1)
            ax2.scatter(np.linspace(epoch-1,epoch,50), np.linspace(xlk_tr_old, losses[3],50),color='red', s=1)
            ax2.set_ylim(bottom=0)
            
            ax3 = plt.subplot(3, 1, 3)
            ax3.scatter(np.linspace(epoch-1,epoch,50), np.linspace(tr_old, losses[0],50),color='blue', s=1)
            ax3.set_ylim(bottom=0)
            plt.savefig(params.model_name + ".png")
            plt.pause(0.05)

        recon_loss_tst_old = recon_loss_tst
        xlk_tr_old_tst = xlk_tr_tst
        done = 1
        re_tr_old = losses[2]
        xlk_tr_old = losses[3]
        tr_old = losses[0]

        if(epoch%params.save_step==0):
            save_model(model, optimizer, epoch,params, "/model_" + str(epoch))
            


        # if(params.disp_flg):
        #     losses_now = [loss.data[0]]#, kl_loss, recon_loss]
        #     if(epoch==0):
        #             losses_epch = losses_update(losses_now)
        #     else:
        #             # print(losses)
        #             for j in range(len(losses)):
        #                 viz.line(X=np.linspace(it2-1,it2,50), Y=np.linspace(losses[j], losses_now[j],50),name=str(j), update='append', win=win)
        #             losses = losses_update(losses_now, losses)
        #     if(it2 % 100 == 0 ):
        #             win = viz.line(X=np.arange(it2, it2 + .1), Y=np.arange(0, .1))
        #     it2 += 1
        if(xlk_tr_tst < xlk_tr_tst_bst):
            xlk_tr_tst_bst = xlk_tr_tst
            print("====== New REGEN-GOAT =====")
            save_model(model,optimizer, epoch,params, "/model_best_test_regen")

        
        out = ""
        if(p_best[0]< p_new[0]):
            p_best = p_new
            print("====== New GOAT =====")
            save_model(model,optimizer, epoch,params, "/model_best_test")
        for i in range(len(p_best)):
            out += str(i) + ":" + str(p_best[i]) + " "
        print(out)
            
        print("="*50)

    plt.show()