import argparse
parser = argparse.ArgumentParser()

# dataset
parser.add_argument('--device', type=str, default='auto', help='device: auto (auto-detect), mps, cuda, or cpu')
parser.add_argument('-id', '--device_id', type=int, default='0', help='device id')

parser.add_argument('-d', '--dataset', type=str, default='eat', help='dataset name')
parser.add_argument('--n_clusters', type=int, default=7, help='cluster number')

# pre-process
parser.add_argument('--n_input', type=int, default=500, help='input feature dimension')

# network
parser.add_argument('-a','--alpha_value', type=float, default=0.2, help='teleport probability')
parser.add_argument('-drop','--drop_edge', type=float, default=0.1, help='drop edge')
parser.add_argument('-g1','--gae_n_enc_1', type=int, default=1000)
parser.add_argument('-g2','--gae_n_enc_2', type=int, default=500)
parser.add_argument('-g3','--gae_n_enc_3', type=int, default=500)
parser.add_argument('-t','--temperature', type=float, default=0.1)
parser.add_argument('--model_type', type=str, default='gae', choices=['gae', 'graphsage'], help='Model type: gae (GCN-based) or graphsage (GraphSAGE-based)')
parser.add_argument('--sage_aggr', type=str, default='mean', choices=['mean', 'max'], help='Aggregation method for GraphSAGE: mean or max')
parser.add_argument('--sage_dropout', type=float, default=0.0, help='Dropout rate for GraphSAGE layers')


# training
parser.add_argument('-e','--epochs', type=int, default=400, help='training epoch') 
parser.add_argument('-s','--seed', type=int, default=0, help='random seed')
parser.add_argument('-eps','--epsilon', type=float, default=0.05, help='regularization parameter for Sinkhorn-Knopp algorithm')
parser.add_argument('-sk','--sinkhorn_iterations', type=int, default=3, help='number of iterations in S-K algorithm')
parser.add_argument('--lam', type=float, default=0.1, help='The hyperparameter of loss')

args = parser.parse_args()
