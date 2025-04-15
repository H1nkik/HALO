import torch
import random
import numpy as np
import opt
from sklearn import metrics
from munkres import Munkres
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score

def setup():
    setup_seed(opt.args.seed)
    if opt.args.dataset == 'eat':
        opt.args.lr = 1e-3
        opt.args.n_clusters = 4
        opt.args.n_input = 203
   
   
    elif opt.args.dataset == "uat":
        opt.args.lr = 7e-4
        opt.args.n_clusters = 4
        opt.args.n_input = 239

    elif opt.args.dataset == "acm": 
        opt.args.n_clusters = 3
        opt.args.lr = 9e-4
        opt.args.n_input = 1870
        opt.args.alpha_value=0.1
        opt.args.sinkhorn_iterations=4
        opt.args.epochs = 100 
        opt.args.epsilon = 0.2
        opt.args.gae_n_enc_1=1024
        opt.args.gae_n_enc_2=300
        opt.args.gae_n_enc_3=60
        
    elif opt.args.dataset == "cornell":
        opt.args.n_clusters = 5
        opt.args.lr = 1e-3
        opt.args.n_input = 1703
        
    elif opt.args.dataset == "amar":
        opt.args.n_clusters = 5
        opt.args.lr = 2e-3
        opt.args.n_input = 300
        opt.args.epochs=100
    elif opt.args.dataset == "sq":
        opt.args.n_clusters = 5
        opt.args.lr = 7e-4
        opt.args.n_input = 2089
        opt.args.epochs=100
    # other new datasets
    else:
        opt.args.lr = 1e-3
        
    print("---------------------")
    print("dataset      : {}".format(opt.args.dataset))
    print("device       : {}".format(opt.args.device))
    print("learning rate: {}".format(opt.args.lr))
    print("lambda       : {}".format(opt.args.lam))
    print("epochs       : {}".format(opt.args.epochs))
    print("---------------------")


def cluster_acc(y_true, y_pred):
    """
    calculate clustering acc and f1-score
    Args:
        y_true: the ground truth
        y_pred: the clustering id

    Returns: acc and f1-score
    """
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    
    if num_class1 != numclass2:
        print('error')
        return
    cost = np.zeros((num_class1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    return acc, f1_macro


def eva(y_true, y_pred, show_details=True):
    """
    evaluate the clustering performance
    Args:
        y_true: the ground truth
        y_pred: the predicted label
        show_details: if print the details
    Returns: None
    """
    
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    if show_details:
            print(':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
                ', f1 {:.4f}'.format(f1))
    return acc, nmi, ari, f1


def load_graph_data(dataset_name, show_details=False):
    load_path = "dataset/" + dataset_name + "/" + dataset_name 
    feat = np.load(load_path+"_feat.npy", allow_pickle=True)
    label = np.load(load_path+"_label.npy", allow_pickle=True) #label start from 0
    adj = np.load(load_path+"_adj.npy", allow_pickle=True)
    cluster_num = len(np.unique(label))
    node_num = feat.shape[0] #
    
    if show_details:
        print("++++++++++++++++++++++++++++++")
        print("---details of graph dataset---")
        print("++++++++++++++++++++++++++++++")
        print("dataset name:   ", dataset_name)
        print("feature shape:  ", feat.shape)
        print("label shape:    ", label.shape)
        print("adj shape:      ", adj.shape)
        print("undirected edge num:   ", int(np.nonzero(adj)[0].shape[0]/2))
        print("Cluster num:           ", cluster_num)
        print("category distribution: ")
        for i in range(max(label)+1):
            print("label", i, end=":")
            print(len(label[np.where(label == i)]))
        print("++++++++++++++++++++++++++++++")

    return feat, label, torch.tensor(adj).float(), node_num, cluster_num


def normalize_adj(adj, self_loop=True, symmetry=False):
    """
    normalize the adj matrix
    :param adj: input adj matrix
    :param self_loop: if add the self loop or not
    :param symmetry: symmetry normalize or not
    :return: the normalized adj matrix
    """
    # add the self_loop
    if self_loop:
        adj_tmp = adj + np.eye(adj.shape[0])
    else:
        adj_tmp = adj

    # calculate degree matrix and it's inverse matrix
    d = np.diag(adj_tmp.sum(0))
    d_inv = np.linalg.inv(d)

    # symmetry normalize: D^{-0.5} A D^{-0.5}
    if symmetry:
        sqrt_d_inv = np.sqrt(d_inv)
        norm_adj = np.matmul(np.matmul(sqrt_d_inv, adj_tmp), sqrt_d_inv)

    # non-symmetry normalize: D^{-1} A
    else:
        norm_adj = np.matmul(d_inv, adj_tmp)
    return norm_adj


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
##### LOSS begin ######
### NT-Xent ###

def calc_loss_a(z_1, z_2, c_1, c_2, temperature=opt.args.temperature, sym=False):
    #norm
    z_1_abs = z_1.norm(dim=1)#+ 1e-8 #avoid norm=0 if + 1e-8
    z_2_abs = z_2.norm(dim=1)#+ 1e-8

    #cos sim
    sim_matrix = torch.einsum('ik,jk->ij', z_1, z_2) / torch.einsum('i,j->ij', z_1_abs, z_2_abs)
    
    #If your GPU > 40G, I recommend this for faster computing:
    A_abs_diff = torch.abs(c_1[:, None, :] - c_2[None, :, :])
    A_sum = A_abs_diff.sum(dim=-1) + 2 # Constant=2
    
    #If your GPU < 40G, i recommend this:
    N = c_1.shape[0]
    # batch_size = 1024
    # A_sum = torch.zeros((N, N), device=opt.args.device)  # 
    # for i in range(0, N, batch_size):
    #     start_idx = i
    #     end_idx = min(i + batch_size, N)  # last batch
    #     c1_batch = c_1[start_idx:end_idx]  # current batch (batch_size, C)
    #     diff_batch = torch.abs(c1_batch[:, None, :] - c_2[None, :, :])  # (batch_size, N, C)
    #     A_sum[i:i + batch_size] = diff_batch.sum(dim=-1) + 2  # compute sub-block A_sum
    
    
    #with torch.autocast(device_type='cuda', dtype=torch.float16):  # to FP16
    #c1_sum = c_1.sum(dim=-1, keepdim=True)
    #c2_sum = c_2.sum(dim=-1, keepdim=True).T
    #min_val = torch.min(c_1[:, None, :], c_2[None, :, :]).sum(dim=-1)
    #A_sum = c1_sum + c2_sum + 2 - 2 * min_val
    diag_mask = torch.eye(A_sum.shape[0], dtype=torch.bool, device= opt.args.device) #mask
    A = torch.where(diag_mask, A_sum + sim_matrix, A_sum - sim_matrix) 
    A_norm = A/A.sum()

    #final sim_matrix
    new_sim_matrix = torch.mul(A_norm, sim_matrix)
    
    final_sim_matrix = torch.exp(new_sim_matrix / temperature)
    #diagonal
    pos_sim = final_sim_matrix[range(N), range(N)]
    
    if sym:
        loss_0 = pos_sim / (final_sim_matrix.sum(dim=0) - pos_sim)
        loss_1 = pos_sim / (final_sim_matrix.sum(dim=1) - pos_sim)
       
        loss_0 = - torch.log(loss_0).mean()
        loss_1 = - torch.log(loss_1).mean()
        loss = (loss_0 + loss_1) / 2.0
    else:
        loss = pos_sim / (final_sim_matrix.sum(dim=1) - pos_sim)
        loss = (- torch.log(loss)).sum()/N
    return loss

def consistent_loss(swap_loss_1,swap_loss_2):
    if opt.args.epsilon==0.2:
        return opt.args.lam*swap_loss_1
    else:
        return opt.args.lam*(swap_loss_1+swap_loss_2)/2 

def swap_loss(p,q): 
    p_log = torch.log(p)
    matrix_mul = torch.mul(q,p_log)
    return -torch.sum(matrix_mul)


#like swav
#@torch.no_grad()
def distributed_sinkhorn(out):
    out = out.T 
    Q = torch.exp(out / opt.args.epsilon).t() 
    B = Q.shape[0]  # number of samples to assign. 
    K = Q.shape[1] # how many prototypes
    
    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    # dist.all_reduce(sum_Q)
    Q = Q/sum_Q

    for it in range(opt.args.sinkhorn_iterations): 
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        # dist.all_reduce(sum_of_rows)
        Q = Q/sum_of_rows
        Q = Q/K

        # normalize each column: total weight per sample must be 1/B
        Q = Q/torch.sum(Q, dim=0, keepdim=True)
        Q = Q/ B

    Q = Q* B # the colomns must sum to 1 so that Q is an assignment
    return Q 

#### Graph data augmentation begin####
def gaussian_noised_feature(X):
    """
    add gaussian noise to the attribute matrix X
    Args:
        X: the attribute matrix
    Returns: the noised attribute matrix X_tilde
    """
    N_1 = torch.Tensor(np.random.normal(1, 0.1, X.shape)).to(opt.args.device)
    N_2 = torch.Tensor(np.random.normal(1, 0.1, X.shape)).to(opt.args.device)
    X_tilde1 = X * N_1
    X_tilde2 = X * N_2
    return X_tilde1, X_tilde2


def diffusion_adj_torch(adj, transport_rate):
    adj_tmp = adj + torch.eye(adj.size(0), device=adj.device)  # add self loop

    # Calculate degree matrix and its inverse
    d = torch.diag(adj_tmp.sum(0))
    d_inv = torch.linalg.inv(d)
    sqrt_d_inv = torch.sqrt(d_inv)

    # Calculate norm adj
    norm_adj = sqrt_d_inv @ adj_tmp @ sqrt_d_inv

    # Calculate graph diffusion
    
    identity = torch.eye(d.size(0), device=adj.device)
    diff_adj = transport_rate * torch.linalg.inv(identity - (1 - transport_rate) * norm_adj)

    return diff_adj


def remove_edge_randomly(A, remove_rate=0.1):
    """
    remove edge randomly
    Args:
        A: the origin adjacency matrix
        remove_rate: the rate of removing linkage relation
    Returns:
        Am: edge-masked adjacency matrix
    """
    # remove edges randomly
    n_node = A.shape[0]
    #A = torch.from_numpy(A)
    for i in range(n_node):
        # find i neighbor
        neighbors = torch.where(A[i] > 0)[0]
        n_remove = int(round(remove_rate * len(neighbors)))
        
        if n_remove > 0:
            # random removal
            remove_indices = np.random.choice(neighbors.cpu().numpy(), n_remove, replace=False)
            A[i, remove_indices] = 0

    # normalize adj
    Anorm = normalize_adj(A, self_loop=True, symmetry=False)
    Anorm = Anorm.to(opt.args.device)
    return Anorm
#### Graph data augmentation end####

def count_parameters(model):
    """
    count the parameters' number of the input model
    Note: The unit of return value is millions(M) if exceeds 1,000,000.
    :param model: the model instance you want to count
    :return: The number of model parameters, in Million (M).
    """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return round(num_params / 1e6, 6)

