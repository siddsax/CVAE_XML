from header import *
from cnn_test import *
from timeit import default_timer as timer

# ---------------------------------------------------------------------------------


def train(x_tr, y_tr, x_te, y_te, x_20, y_20, embedding_weights, params, decoder_word_input=None, decoder_target=None, decoder_word_input_t=None, decoder_target_t=None):

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

	print(model); print("%"*100)
	print("Num Parameters : Encoder {} Classifier {} Variational {} Decoder {}".format(count_parameters(model.encoder), count_parameters(model.classifier), count_parameters(model.variational), count_parameters(model.decoder)))

	optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params.lr)
	# =============================== TRAINING ====================================
	for epoch in range(params.num_epochs):
		for i in range(int(num_mb)):
			# ------------------ Load Batch Data ---------------------------------------------------------
			
			start = timer()
			batch_x, batch_y, decoder_word_input_b, decoder_target_b = load_batch_cnn(x_tr, y_tr, params)
			loading = timer() - start
			# -----------------------------------------------------------------------------------
			# loss, kl_loss, cross_entropy, cross_entropy_y, cross_entropy_y_act = model.forward(
			# 	batch_x, batch_y, decoder_word_input_b, decoder_target_b)
			start = timer()
			loss = model.forward(batch_x, batch_y, decoder_word_input_b, decoder_target_b)
			propogation = timer() - start
			start = timer()
			loss.backward()
			get_loss = timer() - start
			start = timer()
			optimizer.step()
			optim_tm = timer() - start

			print("Times: Loading: {} Propagation: {} Loss: {} Optimization: {}".format(loading, propogation, get_loss, optim_tm))
			exit()

			# CNN_XML
			# Times: Loading: 0.00494694709778 Propagation: 0.160566091537 Loss: 0.0592050552368 Optimization: 0.0538489818573
