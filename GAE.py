import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module,Parameter
import opt

class GNNLayer(Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act = nn.Tanh()

        self.w = Parameter(torch.FloatTensor(out_features, in_features))
        torch.nn.init.xavier_uniform_(self.w)

    def forward(self, features, adj, active):
        if active:
          support = self.act(F.linear(features, self.w)) 
        else:
          support = F.linear(features, self.w)  # add bias
          
        output = torch.mm(adj, support)
        return output

class GAE(nn.Module):
    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, n_input):
        super(GAE, self).__init__()
        self.gnn_1 = GNNLayer(n_input, gae_n_enc_1)
        self.gnn_2 = GNNLayer(gae_n_enc_1, gae_n_enc_2)
        self.gnn_3 = GNNLayer(gae_n_enc_2, gae_n_enc_3)

    def forward(self, x, adj):
        z = self.gnn_1(x, adj, active=True)
        z = self.gnn_2(z, adj, active=True)
        z_igae = self.gnn_3(z, adj, active=False)

        return z_igae

class Cluster_layer(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(Cluster_layer, self).__init__()
        #Same as CC, two MLP relu+softmax
        self.cluster_head =  nn.Sequential(nn.Linear(in_dims, in_dims),
                                nn.ReLU(),
                                nn.Linear(in_dims, out_dims),
                                nn.Softmax(dim=1))

    def forward(self, h):
        c = self.cluster_head(h)
        return  c

#GAE+cluster projection
class GAEC(nn.Module):
    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, n_input):
        super(GAEC, self).__init__()
        self.encoder = GAE(
            gae_n_enc_1= gae_n_enc_1,
            gae_n_enc_2= gae_n_enc_2,
            gae_n_enc_3= gae_n_enc_3,
            n_input=n_input
            )
        self.cluster = Cluster_layer(
            in_dims=gae_n_enc_3,
            out_dims=opt.args.n_clusters)# N*C
        
    def forward(self, x, adj):
        z_igae = self.encoder(x, adj)
        c = self.cluster(z_igae)
        return z_igae, c
