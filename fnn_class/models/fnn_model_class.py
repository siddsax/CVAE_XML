from header import *
import math
import pdb
from models import *
from decoder_classify import *
class fnn_model_class(nn.Module):
    def __init__(self, params):
        super(fnn_model_class, self).__init__()
        self.params = params
        self.encoder = encoder(params)
        self.decoder = decoder(params)
        self.variational = variational(params)
        self.classifier = classifier(params)
        
    def forward(self, batch_x, batch_y=None, test=0):
        # ----------- Encode (X, Y) --------------------------------------------
        z_mean, z_log_var = self.variational(batch_x)
        z = sample_z(z_mean, z_log_var, self.params)
        # ---------------------------------------------------------------
        # ----------- Decode (X, z) --------------------------------------------
        # ------------------ Check for Recon Loss ----------------------------
        if(batch_y is not None):
            Y_sample = self.classifier(batch_x)
            X_sample = self.decoder(z, batch_y)
            lkhood_xy = self.params.loss_fns.logxy_loss(batch_x, X_sample, self.params)
            recon_loss = self.params.loss_fns.cls_loss(batch_y, Y_sample, self.params)
            
            if(test):
                kl_loss = 0.0
                loss = recon_loss + lkhood_xy
            else:
                kl_loss = self.params.loss_fns.kl(z_mean, z_log_var)
                loss = recon_loss + kl_loss + lkhood_xy
                kl_loss = kl_loss.data[0]
            
            recon_loss = recon_loss.data[0]
            lkhood_xy = lkhood_xy.data[0]
            if(test):
                return loss, recon_loss, lkhood_xy, kl_loss, Y_sample.data.cpu().numpy(), X_sample.data.cpu().numpy()
            else:
                return loss, recon_loss, lkhood_xy, kl_loss
        else:
            Y_sample = self.classifier(batch_x)
            X_sample = self.decoder(z, Y_sample)
            lkhood_xy = self.params.loss_fns.logxy_loss(batch_x, X_sample, self.params)

            entropy = self.params.loss_fns.entropy(Y_sample)
            if(test):
                kl_loss = 0.0
                labeled_loss = lkhood_xy
            else:
                kl_loss = self.params.loss_fns.kl(z_mean, z_log_var)
                labeled_loss = kl_loss + lkhood_xy
                kl_loss = kl_loss.data[0]
            loss = labeled_loss #+ entropy
            # loss = torch.mean(torch.sum(Y_sample * labeled_loss, dim=-1)) + entropy
            entropy = entropy.data[0]
            labeled_loss = labeled_loss.data[0]
            return loss, entropy, labeled_loss

            
        
        # if(recon_loss.data[0]<0):
        #     print(recon_loss)
        #     print(Y_sample[0:100])
        #     print(batch_y[0:100])
        #     sys.exit()
        # # ---------------------------------------------------------------------
        # if(math.isnan(kl_loss)):
        #     print(z_var)
        #     # print(np.max(z_var))
        #     pdb.set_trace()
        #     sys.exit()
        # if(math.isnan(recon_loss)):
        #     print("------======----------")
        #     pdb.set_trace()
        # ------------ Loss --------------------------------------------------
        # loss = recon_loss + kl_loss
        # --------------------------------------------------------------------

        # kl_loss = kl_loss.data[0]
        # recon_loss = recon_loss.data[0]
    
