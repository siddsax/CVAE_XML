from header import *
import math
import pdb
class fnn_model_class(nn.Module):
    def __init__(self, params):
        super(fnn_model_class, self).__init__()
        self.params = params
        self.encoder = encoder(params)
        self.decoder = decoder(params)
        
    def forward(self, batch_x, batch_y):
        # ----------- Encode (X, Y) --------------------------------------------
        # inp = torch.cat([X],1).type(params.dtype)
        z_mu, z_var  = self.encoder.forward(batch_x)
        z = sample_z(z_mu, z_var, self.params)
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
        # ---------------------------------------------------------------
        
        # ----------- Decode (X, z) --------------------------------------------
        Y_sample = self.decoder.forward(z_mu)
        recon_loss = self.params.loss_fn(Y_sample, batch_y)
        # ------------------ Check for Recon Loss ----------------------------
        if(recon_loss.data[0]<0):
            print(recon_loss)
            print(Y_sample[0:100])
            print(batch_y[0:100])
            sys.exit()
        # ---------------------------------------------------------------------
        if(math.isnan(kl_loss)):
            print(z_var)
            # print(np.max(z_var))
            pdb.set_trace()
            sys.exit()
        if(math.isnan(recon_loss)):
            print("------======----------")
            pdb.set_trace()
        # ------------ Loss --------------------------------------------------
        loss = recon_loss + kl_loss
        # loss = self.params.beta*recon_loss + kl_loss
        # --------------------------------------------------------------------

        # return loss.view(-1,1), kl_loss.view(-1,1), recon_loss.view(-1,1)
        # return loss, kl_loss, recon_loss
        return loss
    