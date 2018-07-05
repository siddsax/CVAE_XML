from header import *
from cnn_decoder import cnn_decoder
from cnn_encoder import cnn_encoder
from classifier import classifier
from variational import variational
from precision_k import precision_k
from weights_init import weights_init

class cnn_encoder_decoder(nn.Module):
    def __init__(self, params, embedding_weights):
        super(cnn_encoder_decoder, self).__init__()
        self.params = params
        self.embedding_layer = embedding_layer(params, embedding_weights)
        self.encoder = cnn_encoder(params)
        self.variational = variational(params)
        self.classifier = classifier(params)
        self.decoder = cnn_decoder(params)
        
    def forward(self, batch_x, batch_y, decoder_word_input, decoder_target):
        # ----------- Encode (X, Y) --------------------------------------------
        e_emb = self.embedding_layer.forward(batch_x)
        H = self.encoder.forward(e_emb)

        Y = self.classifier(H)
        cross_entropy_y = self.params.loss_fn(Y, batch_y)
 
        if(cross_entropy_y.data[0]<0):
            print(cross_entropy_y)
            print(Y[0:100])
            print(batch_y[0:100])
            sys.exit()

        return cross_entropy_y.view(-1,1)