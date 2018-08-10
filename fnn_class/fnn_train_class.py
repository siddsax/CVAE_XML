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
    best_test_loss = 1e10
    init = 0
    # loss_names = ['lossF', 'lossL', 'recon_loss', 'dist', 'kl_loss', 'zero_dist', 'dist_bce', 'dist_l1', "dist_mse", 'lossU', 'entropy', 'dist', 'kl_loss_ss']
    loss_names = ['lossF', 'lossL', 'recon_loss', 'dist', 'kl_loss', 'zero_dist', 'lossU', 'entropy', 'dist', 'kl_loss_ss']
    model = fnn_model_class(params)
    print(model)
    if not os.path.exists('saved_models/' + params.model_name ):
        os.makedirs('saved_models/' + params.model_name)
    logs = open("saved_models/" + params.model_name + "/logs.txt", 'w+')
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
        if(params.freezing):
            model = load_model(model, params.load_model + "/model_best_test")
        else:
            model, optimizer, init = load_model(model, params.load_model + "/model_best_test", optimizer)
        model.eval()
        best_test_loss,p_new, recon_loss_tst, xlk_tr_tst = test(x_te, y_te, params, model=model, best_test_loss=best_test_loss)
        print("*"*75)
        model.train()

    re_tr_old = 0.0
    xlk_tr_old = 0.0
    tr_old = 0.0
    best_epch_loss = 1e10
    done = 0
    xlk_tr_tst_bst = 1e10
    p_best = np.zeros(5)
    num_mb = np.ceil(params.N/params.mb_size)*params.ratio

    for epoch in range(init, params.num_epochs):
        logs = open("saved_models/" + params.model_name + "/logs.txt", 'a+')
        prem = 0
        for it in range(int(num_mb)):
            if(params.train_labels):
                params.mb_size /= params.ratio
                X, Y = load_data(x_tr, y_tr, params)
                params.mb_size *= params.ratio 
                # lossL, recon_loss, dist, kl_loss, dist_l1, dist_bce, dist_mse = model(X, Y)
                lossL, recon_loss, dist, kl_loss = model(X, Y)
                zero_dist = model.params.loss_fns.logxy_loss(X, Variable(torch.zeros(X.shape).type(model.params.dtype)), model.params).data.cpu().numpy()[0]
                losses_new = [lossL.data[0], recon_loss, dist, kl_loss, zero_dist]#, dist_bce, dist_l1, dist_mse]
            else:
                losses_new = [0.0, 0.0, 0.0, 0.0, 0.0]#, 0.0, 0.0, 0.0]
            if(params.ss):
                dummy = np.zeros(np.shape(x_unl))
                XU, _ = load_data(x_unl, dummy, params) ###### using dummy as dummy
                lossU, entropy, dist, kl_loss_ss = model(XU)
                losses_new += [lossU.data[0], entropy, dist, kl_loss_ss]

            if(params.ss and params.train_labels): 
                lossF = lossL + lossU
            elif(params.train_labels):
                lossF = lossL
            elif(params.ss):
                lossF = lossU
            else:
                print("Error, neither labeled or unlabeled data give")
                exit()
            losses_new = [lossF.data[0]] + losses_new

            # ------------------------ Propogate loss -----------------------------------
            lossF.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), params.clip)
            for p in model.parameters():
                if p.grad is not None:
                    flag = np.isnan(p.grad.data.cpu().numpy()).any()
                    if flag:
                        losses = losses_add(losses_new, losses) if prem else losses_add(losses_new)
                        out = ""
                        for i in range(len(losses_new)):
                            out+= loss_names[i] + ":" + str(np.around(losses_new[i], decimals=4)) + " "
                        print(out)
                        print(" Naans cooking up! ")
                        import pdb
                        pdb.set_trace()
                    flag = np.isinf(p.grad.data.cpu().numpy()).any()
                    if flag:
                        print("Limitless! ")
                        import pdb
                        pdb.set_trace()
            optimizer.step()
            optimizer.zero_grad()

            # ----------------------------------------------------------------------------
            losses = losses_add(losses_new, losses) if prem else losses_add(losses_new)
            if it % max(int(num_mb/12),5) == 0:
                out = ""
                for i in range(len(losses_new)):
                    out+= loss_names[i] + ":" + str(np.around(losses_new[i], decimals=4)) + " "
                print(out)
            prem = 1
        # ------------------- Save Model, Run on test data and find mean loss in epoch ----------------- 
        print("="*50)    
        if (params.justClassify == 0 or epoch % 5 == 0):
            for i in range(len(losses)):
                losses[i] /= num_mb
            if(losses[0]<best_epch_loss):
                best_epch_loss = losses[0]
                save_model(model, optimizer, epoch, params, "/model_best_train")

            out="Model Name :" + params.model_name + " Epoch No: " + str(epoch) + " "
            for i in range(len(losses_new)):
                out+= loss_names[i] + ":" + str(np.around(losses[i], decimals=3)) + " "

            best_test_loss,p_new, recon_loss_tst, xlk_tr_tst = test(x_te, y_te, params, model=model, best_test_loss=best_test_loss)

            if(xlk_tr_tst < xlk_tr_tst_bst):
                xlk_tr_tst_bst = xlk_tr_tst
                print("====== New REGEN-GOAT =====")
                save_model(model,optimizer, epoch,params, "/model_best_test_regen")

            out += "regen best test: " + str(np.around(xlk_tr_tst_bst, decimals=3)) + " "

            print(out)
            out = out + "recon_loss_tst: " + str(np.around(recon_loss_tst, decimals=3)) + " P_1 " + str(p_new[0]) + "\n"
            logs.write(out)
            logs.close()

            out = ""
            if(p_best[0]< p_new[0]):
                p_best = p_new
                print("====== New GOAT =====")
                save_model(model,optimizer, epoch,params, "/model_best_test")
            for i in range(len(p_best)):
                out += str(i) + ":" + str(p_best[i]) + " "
            print(out)

            print("="*50)

            done += 1
            if(epoch%100==0 or done == 3):
                plt.gcf().clear()
                ax1 = plt.subplot(3, 1, 1)
                ax2 = plt.subplot(3, 1, 2)
                ax3 = plt.subplot(3, 1, 3)
                ax1.set_xlabel('Classification Error')
                ax2.set_xlabel('SS-Reconstruction Error')
                ax3.set_xlabel('Training Error')
                ax1.set_ylim(bottom=0, top=1.5*losses[2])
                ax2.set_ylim(bottom=0, top=1.5*losses[3])
                ax3.set_ylim(bottom=0, top=1.5*losses[0])
            if(done>=3):
                ax1.scatter(np.linspace(epoch-1,epoch,50), np.linspace(recon_loss_tst_old, recon_loss_tst,50),color='blue', s=1)
                ax1.scatter(np.linspace(epoch-1,epoch,50), np.linspace(re_tr_old, losses[2],50),color='red', s=1)

                # ax2.scatter(np.linspace(epoch-1,epoch,50), np.linspace(xlk_tr_old_tst, xlk_tr_tst,50),color='blue', s=1)
                # ax2.scatter(np.linspace(epoch-1,epoch,50), np.linspace(xlk_tr_old, losses[3],50),color='red', s=1)
                if(params.ss):
                    ax2.scatter(np.linspace(epoch-1,epoch,50), np.linspace(dist_ss_old, losses[-2],50),color='red', s=1)

                ax3.scatter(np.linspace(epoch-1,epoch,50), np.linspace(tr_old, losses[0],50),color='blue', s=1)
                plt.savefig(params.model_name + ".png")
                plt.pause(0.05)

            recon_loss_tst_old = recon_loss_tst
            xlk_tr_old_tst = xlk_tr_tst
            re_tr_old = losses[2]
            xlk_tr_old = losses[3]
            tr_old = losses[0]
            if(params.ss):
                dist_ss_old = losses[-2]
