from header import *
from encoder import encoder
from decoder import decoder


class fnn_model_gen(nn.Module):
    def __init__(self, params):
        super(fnn_model_gen, self).__init__()
        self.params = params
        self.encoder = encoder(params)
        self.decoder = decoder(params)
        
    def forward(self, batch_x, batch_y):
        # ----------- Encode (X, Y) --------------------------------------------
        # inp = torch.cat([X, Y],1).type(dtype)
        z_mu, z_var  = self.encoder.forward(batch_x)
        z = sample_z(z_mu, z_var, self.params)
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
        # ---------------------------------------------------------------

        # ---------------------------- Decoder ------------------------------------
        # , batch_y
        # z
        inp = torch.cat([z_mu], 1).type(self.params.dtype)
        X_sample = self.decoder.forward(inp)
        recon_loss = self.params.loss_fn(X_sample, batch_x)
        # -------------------------------------------------------------------------

        # ------------------ Check for Recon Loss ----------------------------
        if(recon_loss<0):
            print(recon_loss)
            print(X_sample[0:100])
            print(batch_x[0:100])
            sys.exit()
        # ---------------------------------------------------------------------

        # ------------ Loss --------------------------------------------------
        loss = self.params.beta*recon_loss# + kl_loss
        # --------------------------------------------------------------------

        return loss, kl_loss, recon_loss, X_sample
    