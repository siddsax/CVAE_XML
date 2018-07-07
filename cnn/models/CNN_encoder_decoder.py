from header import *
from cnn_decoder import cnn_decoder
from cnn_encoder import cnn_encoder
from classifier import classifier
from variational import variational
from precision_k import precision_k
from weights_init import weights_init
from timeit import default_timer as timer

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
        Y, H2 = self.classifier(H)
        cross_entropy_y = self.params.loss_fn(Y, batch_y)
        
        start = timer()
        z_mu, z_lvar = self.variational(H)
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_lvar) + z_mu**2 - 1. - z_lvar, 1))
        [batch_size, _] = z_mu.size()
        z = Variable(torch.randn([batch_size, self.params.Z_dim])).type(self.params.dtype_f)
        eps = torch.exp(0.5 * z_lvar).type(self.params.dtype_f)
        z = z * eps + z_mu

 
        decoder_input = self.embedding_layer.forward(decoder_word_input)
        start = timer()
        X_sample = self.decoder.forward(decoder_input, z, H2)
        print("Decoder Time: {}".format(timer()-start))
        X_sample = X_sample.view(-1, self.params.vocab_size)
        print("Time for variational: {}".format(timer()-start))
        cross_entropy = torch.nn.functional.cross_entropy(X_sample, decoder_target)

        # X_sample = self.decoder.forward(decoder_input, z, batch_y) # Supervised loss on encoder
        # X_sample = X_sample.view(-1, self.params.vocab_size)
        # cross_entropy_y_act = torch.nn.functional.cross_entropy(X_sample, decoder_target)
        
        # loss = cross_entropy_y
        # loss = kl_loss + cross_entropy_y# + cross_entropy_y_act
        loss = cross_entropy + kl_loss + cross_entropy_y# + cross_entropy_y_act
        return loss.view(-1,1)