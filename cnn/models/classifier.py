from header import *

# class classifier(nn.Module):
#     def __init__(self, params):
#         super(classifier, self).__init__()
#         self.params = params
#         self.l1 = nn.Linear(params.H_dim, params.h_dim)
#         self.relu = nn.ReLU()
#         self.l2 = nn.Linear(params.h_dim, params.y_dim)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, H):
#         H = self.l1(H)
#         H = self.relu(H)
#         H = self.l2(H)
#         H = self.sigmoid(H)
#         return H




from header import *

class classifier(nn.Module):
    def __init__(self, params):
        super(classifier, self).__init__()
        self.params = params
        # self.l1 = nn.Linear(params.H_dim, params.h_dim)
        self.drp = nn.Dropout(.5)
        self.l1 = nn.Linear(params.H_dim, params.y_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, H):
        H = self.drp(H)
        H = self.l1(H)
        # H = self.relu(H)
        H = self.sigmoid(H)
        # H = self.l2(H)
        return H