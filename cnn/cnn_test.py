from header import *
from collections import OrderedDict

def test_gen(x_te, y_te, embedding_weights, params, model=None):

    if(model==None):
        model = cnn_encoder_decoder(params, embedding_weights)
        model.load_state_dict(torch.load(params.load_model + "/model_best",  map_location=lambda storage, loc: storage))
    
    if(torch.cuda.is_available()):
        model.params.dtype_f = torch.cuda.FloatTensor
        model.params.dtype_i = torch.cuda.LongTensor
        model = nn.DataParallel(model.cuda())
    else:
        model.params.dtype_f = torch.FloatTensor
        model.params.dtype_i = torch.LongTensor
    
    batch_size = x_te.shape[0]


    # for j in range(batch_size):

    #     decoder_word_input_np = np.array([[params.vocabulary[params.go_token]])
        
    #     decoder_word_input = Variable(torch.from_numpy(decoder_word_input_np).long().type(dtype_i))
    #     decoder_input = model.embedding_layer.forward(decoder_word_input)
    #     seed = np.random.normal(size=[1, params.Z_dim])
    #     seed = Variable(torch.from_numpy(seed).float().type(params.dtype_f))
    #     if torch.cuda.is_available():
    #         seed = seed.cuda()
    #         decoder_input = decoder_input.cuda()
    #     result = ''
        
    #     for i in range(params.sequence_length):
    #         logits, _ = model.decoder.forward(decoder_input, seed, y_te)

    #         [_, sl, _] = logits.size()

    #         logits = logits.view(-1, params.vocab_size)
    #         prediction = nn.functional.softmax(logits)
    #         prediction = prediction.view(batch_size, sl, -1)

    #         # take the last word from prefiction and append it to result
    #         word = sample_word_from_distribution(params, prediction.data.cpu().numpy()[j, -1])

    #         if word == params.end_token:
    #             break

    #         result += ' ' + word

    #         word = np.array([[params.vocabulary[word]]])

    #         decoder_word_input_np = np.append(decoder_word_input_np, word, 1)
    #         decoder_word_input = Variable(torch.from_numpy(decoder_word_input_np).long())

    #         if torch.cuda.is_available():
    #             decoder_word_input = decoder_word_input.cuda()

def test_class(x_te, y_te, x_tr, y_tr, embedding_weights, params, model=None):

    
    if(model==None):
        model = cnn_encoder_decoder(params, embedding_weights)
        # original saved file with DataParallel
        state_dict = torch.load(params.load_model + "/model_best", map_location=lambda storage, loc: storage)
        # create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)
        # model.load_state_dict(new_state_dict)
    
    if(torch.cuda.is_available()):
        params.dtype_f = torch.cuda.FloatTensor
        params.dtype_i = torch.cuda.LongTensor
        model = nn.DataParallel(model.cuda())
    else:
        params.dtype_f = torch.FloatTensor
        params.dtype_i = torch.LongTensor
 
    x_tr, y_tr, _, _ = load_batch_cnn(x_tr, y_tr, params, batch=False)

    e_emb = model.embedding_layer.forward(x_tr)
    H = model.encoder.forward(e_emb, y_tr)
    Y = model.classifier(H)
    cross_entropy_y = params.loss_fn(Y, y_tr)
    print('Train Loss; {:.4};'.format(cross_entropy_y.data))#, kl_loss.data, recon_loss.data))

    x_te, y_te, _, _ = load_batch_cnn(x_te, y_te[:,:-1], params, batch=False)
    e_emb2 = model.embedding_layer.forward(x_te)
    H2 = model.encoder.forward(e_emb2, y_te)
    Y2 = model.classifier(H2)
    cross_entropy_y2 = params.loss_fn(Y2, y_te)
    print('Test Loss; {:.4};'.format(cross_entropy_y2.data))#, kl_loss.data, recon_loss.data))
    Y_probabs2 = sparse.csr_matrix(Y2.data)
    sio.savemat('score_matrix.mat' , {'score_matrix': Y_probabs2})

