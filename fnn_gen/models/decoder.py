from header import *

class decoder(torch.nn.Module):

    def __init__(self, params):
        
        super(decoder, self).__init__()

        # self.l0 = nn.Linear(params.Z_dim + params.y_dim, params.h_dim)# + params.y_dim, bias=True)
        self.l0 = nn.Linear(params.Z_dim, params.h_dim)# + params.y_dim, bias=True)
        self.l1 = nn.ReLU()
        self.bn = nn.BatchNorm1d(params.h_dim)# + params.y_dim)

        # self.l2 = nn.Linear(params.h_dim + params.y_dim, params.h_dim)# + params.y_dim, bias=True)
        self.l2 = nn.Linear(params.h_dim, params.h_dim)# + params.y_dim, bias=True)
        self.l3 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(params.h_dim)# + params.y_dim)
        
        # self.l4 = nn.Linear(params.h_dim + params.y_dim, 2*params.h_dim)# + params.y_dim, bias=True)
        self.l4 = nn.Linear(params.h_dim, 2*params.h_dim)# + params.y_dim, bias=True)
        self.l5 = nn.ReLU()
        self.bn3 = nn.BatchNorm1d(2*params.h_dim)# + params.y_dim)

        # self.l6 = nn.Linear(2*params.h_dim + params.y_dim, params.X_dim)#, bias=True)
        self.l6 = nn.Linear(2*params.h_dim, params.X_dim)#, bias=True)

        if(params.fin_layer == "Sigmoid"):
            self.l7 = nn.Sigmoid()
        elif(params.fin_layer == "ReLU"):
            self.l7 = nn.ReLU()
        elif(params.fin_layer == "None"):
            self.l7 = ""
        else:
            print("Error, The final layer given is not defined by me!")
            sys.exit()
        # ------------------------------- Weight init and cleaning ----------------------------
        torch.nn.init.xavier_uniform_(self.l0.weight)
        torch.nn.init.xavier_uniform_(self.l2.weight)
        torch.nn.init.xavier_uniform_(self.l4.weight)
        torch.nn.init.xavier_uniform_(self.l6.weight)

        # -------------------------------------------------------------------------------------


    def forward(self, inputs):
        
        o = self.l0(inputs)
        o = self.l1(o)
        o = self.bn(o)
        
        o = self.l2(o)
        o = self.l3(o)
        o = self.bn2(o)

        o = self.l4(o)
        o = self.l5(o)
        o = self.bn3(o)

        o = self.l6(o)
        if(type(self.l7)!=str):
            o = self.l7(o)

        return o
