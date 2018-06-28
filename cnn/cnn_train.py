from header import *

# ---------------------------------------------------------------------------------

def train(x_tr, y_tr, embedding_weights, params):
    viz = Visdom()
    kl_b = float('Inf')
    lk_b = float('Inf')
    ceya_b = float('Inf')
    cey_b = float('Inf')
    loss_best2 = float('Inf')
    best_epch_loss = float('Inf')
    num_mb = np.ceil(params.N/params.mb_size)
    
    model = cnn_encoder_decoder(params, embedding_weights)
    if(torch.cuda.is_available()):
        print("--------------- Using GPU! ---------")
        model.params.dtype_f = torch.cuda.FloatTensor
        model.params.dtype_i = torch.cuda.LongTensor
        
        model = model.cuda()
    else:
        model.params.dtype_f = torch.FloatTensor
        model.params.dtype_i = torch.LongTensor
        print("=============== Using CPU =========")

    print(model);print("%"*100)
    model = nn.DataParallel(model)
    
    if(len(params.load_model)):
        print(params.load_model)
        model.load_state_dict(torch.load(params.load_model + "/model_best", map_location=lambda storage, loc: storage))

    optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=params.lr)

    print('Boom 5')
    # =============================== TRAINING ====================================
    for epoch in range(params.num_epochs):
        kl_epch = 0
        recon_epch = 0
        cey_epch = 0
        ceya_epch = 0

        for i in range(int(num_mb)):
            # ------------------ Load Batch Data ---------------------------------------------------------
            batch_x, batch_y, decoder_word_input, decoder_target = load_batch_cnn(x_tr, y_tr, params)
            # -----------------------------------------------------------------------------------
            loss, kl_loss, cross_entropy, cross_entropy_y, cross_entropy_y_act = model.forward(batch_x, batch_y, decoder_word_input, decoder_target)

            loss = loss.mean().squeeze()
            kl_loss = kl_loss.mean().squeeze()
            cross_entropy = cross_entropy.mean().squeeze()
            cross_entropy_y = cross_entropy_y.mean().squeeze()
            cross_entropy_y_act = cross_entropy_y_act.mean().squeeze()
            # --------------------------------------------------------------------

            #  --------------------- Print and plot  -------------------------------------------------------------------
            kl_epch += kl_loss.data
            recon_epch += cross_entropy.data
            cey_epch += cross_entropy_y.data
            ceya_epch += cross_entropy_y_act.data
            
            if i % int(num_mb/12) == 0:
                print('Iter-{}; Loss: {:.4}; KL-loss: {:.4} ({:.4}); recons_loss: {:.4} ({:.4}); cross_entropy_y: {:.4} ({:.4}); cross_entropy_y_act: {:.4} ({:.4}); best_loss: {:.4};'.format(i, \
                loss.data, kl_loss.data, kl_b, cross_entropy.data, lk_b, cross_entropy_y.data, cey_b, cross_entropy_y_act.data, ceya_b, loss_best2))

                if not os.path.exists('saved_models/' + params.model_name ):
                    os.makedirs('saved_models/' + params.model_name)
                torch.save(model.state_dict(), "saved_models/" + params.model_name + "/model_best_batch")

                if(loss<loss_best2):
                    loss_best2 = loss.data
                    lk_b = cross_entropy.data
                    kl_b = kl_loss.data
                    cey_b = cross_entropy_y.data
                    ceya_b = cross_entropy_y_act.data

            # -------------------------------------------------------------------------------------------------------------- 
            
            # ------------------------ Propogate loss -----------------------------------
            loss.backward()
            del loss
            optimizer.step()
            optimizer.zero_grad()
            # ----------------------------------------------------------------------------

        kl_epch/= num_mb
        recon_epch/= num_mb
        cey_epch /= num_mb
        ceya_epch /= num_mb
        loss_epch = kl_epch + recon_epch + cey_epch + ceya_epch
        
        if(loss_epch<best_epch_loss):
            best_epch_loss = loss_epch
            if not os.path.exists('saved_models/' + params.model_name ):
                os.makedirs('saved_models/' + params.model_name)
            torch.save(model.state_dict(), "saved_models/" + params.model_name + "/model_best")

        print('End-of-Epoch: Loss: {:.4}; KL-loss: {:.4}; recons_loss: {:.4}; best_loss: {:.4};'.format(loss_epch, kl_epch, recon_epch, best_epch_loss))
        print("="*50)

        if params.save:
            if epoch % params.save_step == 0:
                torch.save(model.state_dict(), "saved_models/" + params.model_name + "/model_" + str(epoch))

        if(params.disp_flg):
            if(epoch==0):
                loss_old = loss_epch
            else:
                viz.line(X=np.linspace(epoch-1,epoch,50), Y=np.linspace(loss_old, loss_epch,50), name='1', update='append', win=win)
                loss_old = loss_epch
            if(epoch % 100 == 0 ):
                win = viz.line(X=np.arange(epoch, epoch + .1), Y=np.arange(0, .1))



