from header import *
from collections import OrderedDict
from sklearn.metrics import log_loss


def test_gen(x_te, y_te, params, model=None, embedding_weights=None):

    if(model==None):
        if(embedding_weights==None):
            print("Error: Embedding weights needed!")
            exit()
        else:
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

def test_class(x_te, y_te, params, model=None, x_tr=None, y_tr=None, embedding_weights=None, verbose=True, save=True ):

    
    if(model==None):
        if(embedding_weights is None):
            print("Error: Embedding weights needed!")
            exit()
        else:
            model = cnn_encoder_decoder(params, embedding_weights)
            # state_dict = torch.load(params.load_model + "/model_best", map_location=lambda storage, loc: storage)
            # new_state_dict = OrderedDict()
            # for k, v in state_dict.items():
            #     name = k[7:]
            #     new_state_dict[name] = v
            # model.load_state_dict(new_state_dict)
            # del new_state_dict
            model.load_state_dict(torch.load(params.load_model + "/model_best_for_test", map_location=lambda storage, loc: storage))
    if(torch.cuda.is_available()):
        params.dtype_f = torch.cuda.FloatTensor
        params.dtype_i = torch.cuda.LongTensor
        model = model.cuda()
    else:
        params.dtype_f = torch.FloatTensor
        params.dtype_i = torch.LongTensor

    model.eval()
    params.mb_size = params.mb_size*5
    if(x_tr is not None and y_tr is not None):
        x_tr, _, _, _ = load_batch_cnn(x_tr, y_tr, params, batch=False)
        Y = np.zeros(y_tr.shape)
        rem = x_tr.shape[0]%params.mb_size
        e_emb = model.embedding_layer.forward(x_tr[-rem:].view(rem, x_te.shape[1]))
        H = model.encoder.forward(e_emb)
        Y[-rem:, :] = model.classifier(H).data
        for i in range(0, x_tr.shape[0] - rem, params.mb_size ):
            print(i)
            e_emb = model.embedding_layer.forward(x_tr[i:i+params.mb_size].view(params.mb_size, x_te.shape[1]))
            H = model.encoder.forward(e_emb)
            Y[i:i+params.mb_size,:] = model.classifier(H).data
    
        cross_entropy_y = log_loss(y_tr, Y)
        print('Train Loss; {:.4};'.format(cross_entropy_y))#, kl_loss.data, recon_loss.data))
        prec_1 = precision_k(y_tr, Y, 1)
        print('Train Loss; {};'.format(prec_1[0]))#, kl_loss.data, recon_loss.data))

    y_te = y_te[:,:-1]
    x_te, _, _, _ = load_batch_cnn(x_te, y_te, params, batch=False)
    Y2 = np.zeros(y_te.shape)
    rem = x_te.shape[0]%params.mb_size
    for i in range(0,x_te.shape[0] - rem,params.mb_size):
        # print(i)
        e_emb2 = model.embedding_layer.forward(x_te[i:i+params.mb_size].view(params.mb_size, x_te.shape[1]))
        H2 = model.encoder.forward(e_emb2)
        Y2[i:i+params.mb_size,:] = model.classifier(H2).data

    if(rem):
        e_emb2 = model.embedding_layer.forward(x_te[-rem:].view(rem, x_te.shape[1]))
        H2 = model.encoder.forward(e_emb2)
        Y2[-rem:,:] = model.classifier(H2).data

    cross_entropy_y2 = log_loss(y_te, Y2) # Reverse of pytorch
    # print('Test Loss; {:.4};'.format(cross_entropy_y2))#, kl_loss.data, recon_loss.data))
    prec_1 = precision_k(y_te, Y2, 1)[0] # Reverse of pytorch
    print('Test Loss; {}; CELoss: {}'.format(prec_1, cross_entropy_y2))#, kl_loss.data, recon_loss.data))
    if(save):
        Y_probabs2 = sparse.csr_matrix(Y2)
        sio.savemat('score_matrix.mat' , {'score_matrix': Y_probabs2})
    params.mb_size = params.mb_size/5
    return prec_1, cross_entropy_y2