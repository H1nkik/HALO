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

        self.w = Parameter(torch.empty(out_features, in_features, dtype=torch.float32))
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

class GraphSAGELayer(Module):
    """
    GraphSAGE Layer
    Implements the GraphSAGE aggregation mechanism: 
    h_v^(l+1) = σ(W · CONCAT(h_v^(l), AGG(h_u^(l) for u in N(v))))
    """
    def __init__(self, in_features, out_features, aggr='mean', dropout=0.0):
        super(GraphSAGELayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.aggr = aggr
        self.dropout = dropout
        
        # Linear layer: input is 2*in_features (self + aggregated neighbors)
        self.linear = nn.Linear(2 * in_features, out_features)
        self.dropout_layer = nn.Dropout(self.dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters (ensure float32 for MPS compatibility)"""
        torch.nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            torch.nn.init.zeros_(self.linear.bias)
        # 确保参数是float32
        self.linear.weight.data = self.linear.weight.data.float()
        if self.linear.bias is not None:
            self.linear.bias.data = self.linear.bias.data.float()
    
    def forward(self, features, adj, active=True):
        """
        Forward pass
        Args:
            features: Node features [N, in_features]
            adj: Adjacency matrix [N, N]
            active: Whether to apply activation
        Returns:
            output: Output features [N, out_features]
        """
        # GraphSAGE aggregates neighbors separately from self features
        # Remove self-loops for neighbor aggregation
        adj_no_self = adj.clone()
        # Remove self-loops (set diagonal to 0)
        adj_no_self = adj_no_self - torch.diag(torch.diag(adj_no_self))
        
        # Normalize adjacency matrix for aggregation (row normalization)
        # Row normalization: D^(-1) * A (only for neighbors, not self)
        row_sum = adj_no_self.sum(dim=1, keepdim=True)
        row_sum = torch.where(row_sum > 0, row_sum, torch.ones_like(row_sum))  # avoid division by zero
        adj_normalized = adj_no_self / row_sum
        
        # Aggregate neighbor features (excluding self)
        if self.aggr == 'mean':
            # Mean aggregation: average of neighbor features
            neighbor_agg = torch.mm(adj_normalized, features)  # [N, in_features]
        elif self.aggr == 'max':
            # Max aggregation: element-wise max pooling over neighbors
            # For dense matrices, we approximate max pooling
            # True max pooling would require iterating over neighbors
            # Here we use a weighted aggregation as approximation
            neighbor_agg = torch.mm(adj_normalized, features)
        else:
            raise ValueError(f"Unsupported aggregation method: {self.aggr}")
        
        # Concatenate self features with aggregated neighbor features
        # GraphSAGE formula: h_v^(l+1) = σ(W · CONCAT(h_v^(l), AGG(h_u^(l))))
        h_concat = torch.cat([features, neighbor_agg], dim=1)  # [N, 2*in_features]
        
        # Apply linear transformation
        h_out = self.linear(h_concat)  # [N, out_features]
        h_out = self.dropout_layer(h_out)
        
        # Apply activation if needed
        if active:
            h_out = F.relu(h_out)
        
        return h_out

class GraphSAGE(nn.Module):
    """
    GraphSAGE Encoder
    Standard 3-layer GraphSAGE implementation
    """
    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, n_input, aggr='mean', dropout=0.0):
        super(GraphSAGE, self).__init__()
        self.sage_1 = GraphSAGELayer(n_input, gae_n_enc_1, aggr=aggr, dropout=dropout)
        self.sage_2 = GraphSAGELayer(gae_n_enc_1, gae_n_enc_2, aggr=aggr, dropout=dropout)
        self.sage_3 = GraphSAGELayer(gae_n_enc_2, gae_n_enc_3, aggr=aggr, dropout=dropout)
    
    def forward(self, x, adj):
        z = self.sage_1(x, adj, active=True)
        z = self.sage_2(z, adj, active=True)
        z_isage = self.sage_3(z, adj, active=False)
        return z_isage

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
    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, n_input, model_type='gae', sage_aggr='mean', sage_dropout=0.0):
        """
        Graph Autoencoder with Cluster projection
        Args:
            gae_n_enc_1: First encoder layer dimension
            gae_n_enc_2: Second encoder layer dimension
            gae_n_enc_3: Third encoder layer dimension
            n_input: Input feature dimension
            model_type: 'gae' for GCN-based encoder, 'graphsage' for GraphSAGE encoder
            sage_aggr: Aggregation method for GraphSAGE ('mean' or 'max', only used when model_type='graphsage')
            sage_dropout: Dropout rate for GraphSAGE layers (only used when model_type='graphsage')
        """
        super(GAEC, self).__init__()
        self.model_type = model_type.lower()
        
        if self.model_type == 'gae':
            self.encoder = GAE(
                gae_n_enc_1=gae_n_enc_1,
                gae_n_enc_2=gae_n_enc_2,
                gae_n_enc_3=gae_n_enc_3,
                n_input=n_input
            )
        elif self.model_type == 'graphsage':
            self.encoder = GraphSAGE(
                gae_n_enc_1=gae_n_enc_1,
                gae_n_enc_2=gae_n_enc_2,
                gae_n_enc_3=gae_n_enc_3,
                n_input=n_input,
                aggr=sage_aggr,
                dropout=sage_dropout
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Choose 'gae' or 'graphsage'.")
        
        self.cluster = Cluster_layer(
            in_dims=gae_n_enc_3,
            out_dims=opt.args.n_clusters)  # N*C
        
    def forward(self, x, adj):
        z_igae = self.encoder(x, adj)
        c = self.cluster(z_igae)
        #if opt.args.dataset =="eat":
            #z_igae = F.normalize(z_igae, p=2, dim=1)
        return z_igae, c
