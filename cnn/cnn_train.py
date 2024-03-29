from header import *
from cnn_test import *

# ---------------------------------------------------------------------------------

def train(x_tr, y_tr, x_te, y_te, x_20, y_20, embedding_weights, params):
		
	viz = Visdom()
	kl_b = float('Inf')
	lk_b = float('Inf')
	ceya_b = float('Inf')
	cey_b = float('Inf')
	loss_best2 = float('Inf')
	best_epch_loss = float('Inf')
	best_test_loss = float('Inf')
	avg_grad = 1e10
	max_grad = 0
	best_test_acc = 0

	num_mb = np.ceil(params.N/params.mb_size)
	num_mb_2 = np.ceil(x_20.shape[0]/params.mb_size)
	
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
	# model = nn.DataParallel(model)
	
	if(len(params.load_model)):
		print(params.load_model)
		model.load_state_dict(torch.load(params.load_model + "/model_best_batch", map_location=lambda storage, loc: storage))

	optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=params.lr)
	# if(len(params.load_model)):
	# 	print(params.load_model)
	# 	checkpoint = torch.load(params.load_model + "/model_best_batch", map_location=lambda storage, loc: storage)
	# 	model.load_state_dict(checkpoint['state_dict'])
	# 	optimizer.load_state_dict(checkpoint['optimizer'])
	# 	init = checkpoint['epoch']
	# else:
	# 	init = 0
	print('Boom 5')
	iteration = 0
	# =============================== TRAINING ====================================
	for epoch in range(params.num_epochs):
		kl_epch = 0.0
		recon_epch = 0.0
		cey_epch = 0.0
		ceya_epch = 0.0

		for i in range(int(num_mb)):
			# ------------------ Load Batch Data ---------------------------------------------------------
			if(params.dataset_gpu):
				batch_x, batch_y, decoder_word_input_b, decoder_target_b = load_batch_cnn(x_tr, y_tr, params, decoder_word_input=decoder_word_input, decoder_target=decoder_target)
			else:
				batch_x, batch_y, decoder_word_input_b, decoder_target_b = load_batch_cnn(x_tr, y_tr, params)
			# -----------------------------------------------------------------------------------
			# loss = model.forward(batch_x, batch_y, decoder_word_input, decoder_target)
			loss, kl_loss, cross_entropy, cross_entropy_y, cross_entropy_y_act = model.forward(batch_x, batch_y, decoder_word_input_b, decoder_target_b)

			loss = loss.mean().squeeze()
			kl_loss = kl_loss.mean().squeeze()
			cross_entropy = cross_entropy.mean().squeeze()
			cross_entropy_y = cross_entropy_y.mean().squeeze()
			cross_entropy_y_act = cross_entropy_y_act.mean().squeeze()
			# --------------------------------------------------------------------

			# ###########################
			batch_x, batch_y, _, _ = load_batch_cnn(x_20, y_20, params, batch_size=2)
			e_emb = model.embedding_layer.forward(batch_x)
			H = model.encoder.forward(e_emb)
			Y = model.classifier(H)
			cross_entropy_y = model.params.loss_fn(Y, batch_y)
			loss = loss + cross_entropy_y
			# ###########################

			#  --------------------- Print and plot  -------------------------------------------------------------------
			kl_epch += kl_loss.data
			recon_epch += cross_entropy.data
			cey_epch += cross_entropy_y.data
			ceya_epch += cross_entropy_y_act.data
			
			if i % int(num_mb/12) == 0:
				print('Iter-{}; Loss: {:.4}; KL-loss: {:.4} ({:.4}); recons_loss: {:.4} ({:.4}); cross_entropy_y: {:.4} ({:.4}); cross_entropy_y_act: {:.4} ({:.4}); best_loss: {:.4}; max_grad: {}'.format(i, \
				loss.data, kl_loss.data, kl_b, cross_entropy.data, lk_b, cross_entropy_y.data, cey_b, cross_entropy_y_act.data, ceya_b, loss_best2, max_grad))

				if not os.path.exists('saved_models/' + params.model_name ):
					os.makedirs('saved_models/' + params.model_name)
				state = {
					'epoch' : epoch,
					'state_dict' : model.state_dict(),
					'optimizer' : optimizer.state_dict()
				}
				torch.save(state, "saved_models/" + params.model_name + "/model_best_batch")

		
				if(loss<loss_best2):
					loss_best2 = loss.data
					lk_b = cross_entropy.data
					kl_b = kl_loss.data
					cey_b = cross_entropy_y.data
					ceya_b = cross_entropy_y_act.data

			# -------------------------------------------------------------------------------------------------------------- 
			
			# ------------------------ Propogate loss -----------------------------------
			loss.backward()
			loss = loss.data
			#optimizer.step()
			torch.nn.utils.clip_grad_norm(model.parameters(), params.clip)
			for p in model.parameters():
				if(p.grad is not None):
					p.data.add_(-params.lr, p.grad.data)
			optimizer.zero_grad()
			# ----------------------------------------------------------------------------
			if(params.disp_flg):
				if(iteration==0):
					loss_old = loss
					# loss_old_t = test_loss
				else:
					viz.line(X=np.linspace(iteration-1,iteration,50), Y=np.linspace(loss_old, loss,50), update='append', win=win)
					# viz.line(X=np.linspace(iteration-1,iteration,50), Y=np.linspace(loss_old_t, test_loss,50), name='2', update='append', win=win)
					loss_old = loss
					# loss_old_t = test_loss
				if(iteration % 100 == 0 ):
					win = viz.line(X=np.arange(iteration, iteration + .1), Y=np.arange(0, .1))
			iteration +=1

			if(epoch==0):
				break


		kl_epch/= num_mb
		recon_epch/= num_mb
		cey_epch /= num_mb
		ceya_epch /= num_mb
		loss_epch = kl_epch + recon_epch + cey_epch + ceya_epch
		
		if(loss_epch<best_epch_loss):
			best_epch_loss = loss_epch
			if not os.path.exists('saved_models/' + params.model_name ):
				os.makedirs('saved_models/' + params.model_name)
			state = {
				'epoch' : epoch,
				'state_dict' : model.state_dict(),
				'optimizer' : optimizer.state_dict()
			}
			torch.save(state, "saved_models/" + params.model_name + "/model_best_epoch")

		print('End-of-Epoch: Loss: {:.4}; KL-loss: {:.4}; recons_loss: {:.4}; best_loss: {:.4};'.format(loss_epch, kl_epch, recon_epch, best_epch_loss))
	
		test_prec_acc, test_ce_loss = test_class(x_te, y_te, params, model=model, verbose=False, save=False)
		model.train()
		
		if(test_prec_acc > best_test_acc):
			best_test_loss = test_ce_loss
			best_test_acc = test_prec_acc
			print("This acc is better than the previous recored test acc:- {} ; while CELoss:- {}".format(best_test_acc, best_test_loss))
			if not os.path.exists('saved_models/' + params.model_name ):
				os.makedirs('saved_models/' + params.model_name)
			state = {
				'epoch' : epoch,
				'state_dict' : model.state_dict(),
				'optimizer' : optimizer.state_dict()
			}
			torch.save(state, "saved_models/" + params.model_name + "/model_best_test")

		
		print("="*50)

		if params.save:
			if epoch % params.save_step == 0:
				state = {
					'epoch' : epoch,
					'state_dict' : model.state_dict(),
					'optimizer' : optimizer.state_dict()
				}
				torch.save(state, "saved_models/" + params.model_name + "/model_" + str(epoch))

		# if(params.disp_flg):
		#     if(epoch==0):
		#         loss_old = loss_epch
		#         loss_old_t = test_loss
		#     else:
		#         viz.line(X=np.linspace(epoch-1,epoch,50), Y=np.linspace(loss_old, loss_epch,50), name='1', update='append', win=win)
		#         viz.line(X=np.linspace(epoch-1,epoch,50), Y=np.linspace(loss_old_t, test_loss,50), name='2', update='append', win=win)
		#         loss_old = loss_epch
		#         loss_old_t = test_loss
		#     if(epoch % 100 == 0 ):
		#         win = viz.line(X=np.arange(epoch, epoch + .1), Y=np.arange(0, .1))



