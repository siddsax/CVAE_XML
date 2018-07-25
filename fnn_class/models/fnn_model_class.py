from header import *
import math
import pdb
from models import *
from decoder_classify import *

def isnan(tensor):
    flag = (tensor != tensor).any()
    return flag
class fnn_model_class(nn.Module):
    def __init__(self, params):
        super(fnn_model_class, self).__init__()
        self.params = params
        self.encoder = encoder(params)
        self.decoder = decoder(params)
        self.variational = variational(params)
        self.classifier = classifier(params)
        self.iter = 0

    def forward(self, batch_x, batch_y=None, test=0):

        if(batch_y is not None):        
            z_mean, z_log_var = self.variational(batch_x, batch_y, self.decoder.emb_layer)
            z = sample_z(z_mean, z_log_var, self.params)
            Y_sample = self.classifier(batch_x)
            X_sample = self.decoder(z, batch_y)
            dist = self.params.loss_fns.logxy_loss(batch_x, X_sample, self.params)
            recon_loss = self.params.loss_fns.cls_loss(batch_y, Y_sample, self.params)
            
            if(test):
                kl_loss = 0.0
                loss = dist + recon_loss
                z_mean, z_log_var = self.variational(batch_x, Y_sample, self.decoder.emb_layer)
                z = sample_z(z_mean, z_log_var, self.params)
                X_sample_from_pred_y = self.decoder(z, Y_sample)
                dist_from_pred_y = self.params.loss_fns.logxy_loss(batch_x, X_sample_from_pred_y, self.params)
                dist_from_pred_y = dist_from_pred_y.data[0]
            else:
                self.iter +=1
                beta = 0#min(1, self.iter/1000000.0)
                kl_loss = self.params.loss_fns.kl(z_mean, z_log_var)
                loss = recon_loss + beta*kl_loss + dist
                import pdb
                if isnan(kl_loss) and isnan(dist):
                    print("kl_loss and dist are nan")
                    pdb.set_trace()
                elif isnan(dist):
                    print("dist")
                    pdb.set_trace()
                elif isnan(kl_loss):
                    print("kl")
                    pdb.set_trace()
                
                kl_loss = kl_loss.data[0]
            
            recon_loss = recon_loss.data[0]
            dist = dist.data[0]
            if(test):
                return loss, recon_loss, dist, dist_from_pred_y, kl_loss, Y_sample.data.cpu().numpy(), X_sample.data.cpu().numpy(), X_sample_from_pred_y.data.cpu().numpy()
            else:
                return loss, recon_loss, dist, kl_loss
        else:
            Y_sample = self.classifier(batch_x)
            z_mean, z_log_var = self.variational(batch_x, Y_sample, self.decoder.l0)
            z = sample_z(z_mean, z_log_var, self.params)
            X_sample = self.decoder(z, Y_sample)
            dist = self.params.loss_fns.logxy_loss(batch_x, X_sample, self.params)

            entropy = self.params.loss_fns.entropy(Y_sample)
            if(test):
                kl_loss = 0.0
                labeled_loss = dist
            else:
                kl_loss = self.params.loss_fns.kl(z_mean, z_log_var)
                labeled_loss = kl_loss + dist
                kl_loss = kl_loss.data[0]
            loss = labeled_loss #+ entropy
            # loss = torch.mean(torch.sum(Y_sample * labeled_loss, dim=-1)) + entropy
            entropy = entropy.data[0]
            labeled_loss = labeled_loss.data[0]
            return loss, entropy, labeled_loss
