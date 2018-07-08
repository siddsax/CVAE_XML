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
        self.ce = torch.nn.CrossEntropyLoss()
        
    def forward(self, batch_x, batch_y, decoder_word_input, decoder_target):
        # ----------- Encode (X, Y) --------------------------------------------
        e_emb = self.embedding_layer.forward(batch_x)
        H = self.encoder.forward(e_emb)
        Y, H2 = self.classifier(H)
        cross_entropy_y = self.params.loss_fn(Y, batch_y)

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
        torch.cuda.synchronize()
        print("Decoder Time: {}".format(timer()-start))
        X_sample = X_sample.view(-1, self.params.vocab_size)
        cross_entropy = self.ce(X_sample, decoder_target)

        # X_sample = self.decoder.forward(decoder_input, z, batch_y) # Supervised loss on encoder
        # X_sample = X_sample.view(-1, self.params.vocab_size)
        # cross_entropy_y_act = torch.nn.functional.cross_entropy(X_sample, decoder_target)
        
        # loss = cross_entropy_y
        # loss = kl_loss + cross_entropy_y# + cross_entropy_y_act
        loss = cross_entropy + kl_loss + cross_entropy_y# + cross_entropy_y_act
        return loss.view(-1,1)

# Time taken in layer 0 in decoder is 0.00420188903809
# Time taken in layer 1 in decoder is 0.00259590148926
# Time taken in layer 2 in decoder is 0.00290894508362
# FC Layer in Decoder: 0.040727853775
# Times are s1:-6.91413879395e-06 s2:-4.50611114502e-05 sx:-1.90734863281e-05 sx2:0 s3:0 s4:-0.000598907470703 s5:-0.00104999542236 s6:-0.000545978546143 s7:-1.90734863281e-05
# Decoder Time: 0.0527701377869
# Times: Loading: 0.00157809257507 Propagation: 0.0669059753418 Loss: 0.109215974808 Optimization: 0.00588798522949

# Times: Loading: 0.00470304489136 Propagation: 1.59442305565 Loss: 0.0100197792053 Optimization: 0.00501489639282
# Times: Loading: 0.00396990776062 Propagation: 0.00341200828552 Loss: 0.00487303733826 Optimization: 0.00326704978943
# Times: Loading: 0.00217700004578 Propagation: 0.00312519073486 Loss: 0.00459885597229 Optimization: 0.00327587127686
# Times: Loading: 0.00158286094666 Propagation: 0.00313401222229 Loss: 0.00460696220398 Optimization: 0.00326299667358
# Times: Loading: 0.00159096717834 Propagation: 0.00312399864197 Loss: 0.0046021938324 Optimization: 0.0032639503479
# Times: Loading: 0.00157117843628 Propagation: 0.00312781333923 Loss: 0.00460696220398 Optimization: 0.00325918197632
