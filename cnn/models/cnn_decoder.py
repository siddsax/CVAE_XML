from header import *
from weights_init import weights_init
from timeit import default_timer as timer

class cnn_decoder(nn.Module):
    def __init__(self, params):
        super(cnn_decoder, self).__init__()
        self.params = params
        self.out_size = self.params.decoder_kernels[-1][0]
        
        self.bn_1 = nn.BatchNorm1d(self.params.sequence_length + 1)
        if(params.dropouts):
            self.drp = nn.Dropout(p=.25)
            self.drp_7 = nn.Dropout(p=.7)
        self.conv_layers = nn.ModuleList()
        self.bn_x = nn.ModuleList()
        self.relu = nn.ReLU()
        for layer in range(len(params.decoder_kernels)):
            [out_chan, in_chan, width] = params.decoder_kernels[layer]
            layer = nn.Conv1d(in_chan, out_chan, width,
                         dilation=self.params.decoder_dilations[layer],
                         padding=self.params.decoder_paddings[layer])
            weights_init(layer.weight)
            bn_layer = nn.BatchNorm1d(out_chan)
            self.conv_layers.append(layer)
            self.bn_x.append(bn_layer)
        
        # self.bn_2 = nn.BatchNorm1d(self.out_size)
        self.fc = nn.Linear(self.out_size, self.params.vocab_size)
        weights_init(self.fc.weight)
        
    # def forward(self, decoder_input, z, batch_y):
    #     [batch_size, seq_len, _] = decoder_input.size()
    #     z = torch.cat([z, batch_y], 1)
    #     z = z.view(batch_size, 1,self.params.Z_dim + self.params.H_dim)
    #     z = z.expand(-1,seq_len,-1)
    #     x = torch.cat([decoder_input, z], 2)
    #     x = x.transpose(1, 2).contiguous()
    #     # if(self.params.dropouts):
    #     #     x = self.drp(x)
    #     torch.cuda.synchronize()
    #     start = timer()
    #     for layer in range(len(self.params.decoder_kernels)):
    #         x = self.conv_layers[layer](x)
    #         x_width = x.size()[2]
    #         x = x[:, :, :(x_width - self.params.decoder_paddings[layer])].contiguous()
    #         x = self.relu(x)
    #         print("Time taken in layer {} in decoder is {}".format(layer,timer()-start))
    #         torch.cuda.synchronize()
    #         start  =  timer()
    #         # x = self.bn_x[layer](x)
    #         # if(self.params.dropouts):
    #         #     x = self.drp_7(x)
        
    #     x = x.transpose(1, 2).contiguous()
    #     if(self.params.multi_gpu):
    #         x = x.cuda(2)
    #         x = self.fc(x)#.cuda(1)
    #     else:
    #         x = self.fc(x)
    #     print("FC Layer in Decoder: {}".format(timer()- start))
    #     x = x.view(-1, seq_len, self.params.vocab_size)
    #     return x

    def forward(self, decoder_input, z, batch_y):
        torch.cuda.synchronize()
        s1 = timer()
        [batch_size, seq_len, _] = decoder_input.size()
        torch.cuda.synchronize()
        s1 = s1 - timer()
        s2 = timer()
        z = torch.cat([z, batch_y], 1)
        torch.cuda.synchronize()
        s2 = s2 - timer()
        sx = timer()
        z = z.view(batch_size, 1,self.params.Z_dim + self.params.H_dim)
        z = z.expand(-1,seq_len,-1)
        torch.cuda.synchronize()
        sx = sx - timer()
        sx2 = 0#timer()
        s3 = 0
        s4 = timer()
        x = torch.cat([decoder_input, z], 2)
        torch.cuda.synchronize()
        s4 = s4 - timer()
        s5 = timer()
        x = x.transpose(1, 2).contiguous()
        torch.cuda.synchronize()
        s5 = s5 - timer()
        # if(self.params.dropouts):
        #     x = self.drp(x)
        start = timer()
        for layer in range(len(self.params.decoder_kernels)):
            x = self.conv_layers[layer](x)
            x_width = x.size()[2]
            x = x[:, :, :(x_width - self.params.decoder_paddings[layer])].contiguous()
            x = self.relu(x)
            torch.cuda.synchronize()
            print("Time taken in layer {} in decoder is {}".format(layer,timer()-start))
            start  =  timer()
            # x = self.bn_x[layer](x)
            # if(self.params.dropouts):
            #     x = self.drp_7(x)
        
        s6 = timer()
        x = x.transpose(1, 2).contiguous()
        torch.cuda.synchronize()
        s6 = s6 - timer()
        start = timer()
        if(self.params.multi_gpu):
            x = x.cuda(2)
            x = self.fc(x)#.cuda(1)
        else:
            x = self.fc(x)
        torch.cuda.synchronize()
        print("FC Layer in Decoder: {}".format(timer()- start))
        s7 = timer()
        x = x.view(-1, seq_len, self.params.vocab_size)
        torch.cuda.synchronize()
        s7 = s7 - timer()
        print("Times are s1:{} s2:{} sx:{} sx2:{} s3:{} s4:{} s5:{} s6:{} s7:{}".format(s1, s2, sx, sx2, s3, s4, s5, s6, s7))
        return x
