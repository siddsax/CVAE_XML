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

        torch.cuda.synchronize()
        start = timer()
        z_mu, z_lvar = self.variational(H)
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_lvar) + z_mu**2 - 1. - z_lvar, 1))
        [batch_size, _] = z_mu.size()
        z = Variable(torch.randn([batch_size, self.params.Z_dim])).type(self.params.dtype_f)
        eps = torch.exp(0.5 * z_lvar).type(self.params.dtype_f)
        z = z * eps + z_mu

 
        decoder_input = self.embedding_layer.forward(decoder_word_input)
        torch.cuda.synchronize()
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

# Times: Loading: 0.00471496582031 Propagation: 1.57417678833 Loss: 0.00685596466064 Optimization: 0.00275301933289
# Times: Loading: 0.00505685806274 Propagation: 0.00342202186584 Loss: 0.00170612335205 Optimization: 0.000565052032471
# Times: Loading: 0.00346422195435 Propagation: 0.00313901901245 Loss: 0.000901937484741 Optimization: 0.000573873519897
# Times: Loading: 0.00344896316528 Propagation: 0.00313091278076 Loss: 0.000900983810425 Optimization: 0.000573873519897
# Times: Loading: 0.00345206260681 Propagation: 0.00329184532166 Loss: 0.000904083251953 Optimization: 0.000581026077271
# Times: Loading: 0.00344276428223 Propagation: 0.0031270980835 Loss: 0.000897884368896 Optimization: 0.000609874725342