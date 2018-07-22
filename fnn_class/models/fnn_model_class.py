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
        self.iter = 0
        
    def forward(self, batch_x, batch_y=None, test=0):

        if(batch_y is not None):
            z_mean, z_log_var = self.variational(batch_x, batch_y, self.decoder.emb_layer)
            z = sample_z(z_mean, z_log_var, self.params)
            if(self.params.compress == 0):
                Y_sample = self.classifier(batch_x)
            X_sample = self.decoder(z, batch_y)
            dist = self.params.loss_fns.logxy_loss(batch_x, X_sample, self.params)
            if(self.params.compress == 0):
                recon_loss = self.params.loss_fns.cls_loss(batch_y, Y_sample, self.params)
            
            if(test):
                kl_loss = 0.0
                if(self.params.compress == 0):
                    loss = dist + recon_loss
                    z_mean, z_log_var = self.variational(batch_x, Y_sample, self.decoder.emb_layer)
                    z = sample_z(z_mean, z_log_var, self.params)
                    X_sample_from_pred_y = self.decoder(z, Y_sample)
                    dist_from_pred_y = self.params.loss_fns.logxy_loss(batch_x, X_sample_from_pred_y, self.params)
                    dist_from_pred_y = dist_from_pred_y.data[0]
                    recon_loss = recon_loss.data[0]
                else:
                    loss = dist
                    dist_from_pred_y = 0.0
                    recon_loss = 0.0
                    
            else:
                self.iter +=1
                beta = min(1, self.iter/1000000.0)
                # gamma = min(1, self.iter/1000000.0)
                kl_loss = self.params.loss_fns.kl(z_mean, z_log_var)
                if(self.params.compress):
                    loss = beta*kl_loss + dist
                    recon_loss = 0.0
                else:
                    loss = beta*kl_loss + dist #+ *recon_loss
                    recon_loss = recon_loss.data[0]
                    
                kl_loss = kl_loss.data[0]
            
            dist = dist.data[0]
            if(test):
                if(self.params.compress):
                    return loss, recon_loss, dist, dist_from_pred_y, kl_loss, 0, X_sample.data.cpu().numpy(), 0
                else:
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
