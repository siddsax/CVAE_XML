from header import *
import math
import pdb
from models import *

def isnan(tensor):
    if isinstance(tensor, float):
        return False
    else:
        flag = (tensor != tensor).any()
    return flag
class fnn_model_class(nn.Module):
    def __init__(self, params):
        super(fnn_model_class, self).__init__()
        self.params = params
        if(params.decoderRes):
            self.decoder = decoderRes(params)
            self.variational = variationalRes(params)
        else:
            self.decoder = decoder(params)
            self.variational = variational(params)

        self.classifier = classifier(params)
        self.beta = 0#.1
        self.gamma = 1.0
        if(params.freezing):
            for param in self.variational.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False

    def forward(self, batch_x, batch_y=None, test=0):

        if(batch_y is not None):
            z_mean, z_log_var = self.variational(batch_x, batch_y)#, self.decoder.emb_layer)
            # z = sample_z(z_mean, z_log_var, self.params)
            Y_sample = self.classifier(batch_x)
            if(self.params.justClassify == 0):
                X_sample = self.decoder(z_mean, batch_y)
                dist = self.params.loss_fns.logxy_loss(batch_x, X_sample, self.params, f=self.params.loss_type)

                dist_l1 = self.params.loss_fns.logxy_loss(batch_x, X_sample, self.params, f=0).data.cpu().numpy()[0]
                dist_bce = -1.0#self.params.loss_fns.logxy_loss(batch_x, X_sample, self.params, f=1).data.cpu().numpy()[0]
                dist_mse = self.params.loss_fns.logxy_loss(batch_x, X_sample, self.params, f=2).data.cpu().numpy()[0]
            else:
                X_sample = Variable(torch.randn(batch_x.shape[0], self.params.X_dim).type(self.params.dtype))
                dist_l1 = -1.0
                dist_bce = -1.0
                dist_mse = -1.0
                dist = 0.0
            recon_loss = self.params.loss_fns.cls_loss(batch_y, Y_sample, self.params)

            if(test):
                kl_loss = 0.0
                loss = dist + recon_loss
                Y_sample_sp = gumbel_multiSample(self.params, Y_sample, self.params.temperature)
                z_mean, z_log_var = self.variational(batch_x, Y_sample_sp)#, self.decoder.emb_layer)
                # z = sample_z(z_mean, z_log_var, self.params)
                X_sample_from_pred_y = self.decoder(z_mean, Y_sample_sp)
                dist_from_pred_y = self.params.loss_fns.logxy_loss(batch_x, X_sample_from_pred_y, self.params, f=self.params.loss_type)
                dist_from_pred_y = dist_from_pred_y.data[0]
                import pdb
                if isnan(dist):
                    print("dist")
                    pdb.set_trace()
            else:
                kl_loss = self.beta#*self.params.loss_fns.kl(z_mean, z_log_var)
                if(self.params.justClassify == 0):
                    loss = recon_loss + kl_loss + dist
                    if isnan(dist):
                        print("dist")
                        pdb.set_trace()
                else:
                    loss = recon_loss + kl_loss

            recon_loss = recon_loss.data[0]
            if(self.params.justClassify == 0):
                dist = dist.data[0]

            if(test):
                return loss, recon_loss, dist, dist_from_pred_y, kl_loss, Y_sample.data.cpu().numpy(), X_sample.data.cpu().numpy(), X_sample_from_pred_y.data.cpu().numpy(), dist_l1, dist_bce, dist_mse
            else:
                return loss, recon_loss, dist, kl_loss, dist_l1, dist_bce, dist_mse
        else:
            Y_sample = self.classifier(batch_x)
            Y_sample = gumbel_multiSample(self.params, Y_sample, self.params.temperature)
            one_label = Y_sample[0].data.cpu().numpy()
            np.savetxt("one_label.csv", one_label, delimiter=",")
            z_mean, z_log_var = self.variational(batch_x, Y_sample)#, self.decoder.l0)
            # z = sample_z(z_mean, z_log_var, self.params)
            X_sample = self.decoder(z_mean, Y_sample)
            dist = self.params.loss_fns.logxy_loss(batch_x, X_sample, self.params, f=self.params.loss_type)

            entropy = self.params.loss_fns.entropy(Y_sample)
            if(test):
                kl_loss = 0.0
                labeled_loss = dist
            else:
                kl_loss = self.beta#*self.params.loss_fns.kl(z_mean, z_log_var)
                labeled_loss = dist# + kl_loss
                # kl_loss = kl_loss.data[0]
                dist = dist.data[0]
            loss = labeled_loss #+ entropy
            entropy = entropy.data[0]
            labeled_loss = labeled_loss.data[0]
            return loss, entropy, dist, kl_loss
