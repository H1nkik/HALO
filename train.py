from opt import args
from utils import * 
from torch.optim import Adam
import opt as opt

def Train_gae(model, data, adj, label):
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    for epoch in range(args.epochs):
        torch.cuda.empty_cache() #used in multi-gpus server

        X_tilde1, X_tilde2 = gaussian_noised_feature(data)
        X_tilde1 = X_tilde1.float() #N*d 
        X_tilde2 = X_tilde2.float()
  
        adj_diffusion = diffusion_adj_torch(adj,  transport_rate=args.alpha_value)
        adj_edge = remove_edge_randomly(adj, remove_rate = args.drop_edge) #torch
        adj_diffusion = adj_diffusion.float().to(args.device)
        adj_edge = adj_edge.float()

        z_igae_1, c_1 = model(X_tilde1, adj_diffusion)  
        z_igae_2, c_2 = model(X_tilde2, adj_edge) 
        
        model.train()
        model.zero_grad() 
        loss_1 = (calc_loss_a(z_igae_1, z_igae_2,c_1,c_2)+calc_loss_a(z_igae_2, z_igae_1,c_2,c_1))/2 #INT-Xent
        #loss_1 = (calc_loss(z_igae_1, z_igae_2)+calc_loss(z_igae_2, z_igae_1))/2 #NT-Xent
        new_c_1 = distributed_sinkhorn(c_1)
        new_c_2 = distributed_sinkhorn(c_2)
        
        swap_loss_1 = swap_loss(new_c_1,new_c_2) # original p(c|z_ij)
        swap_loss_2 = swap_loss(new_c_2,new_c_1) 
         
        loss_2 = consistent_loss(swap_loss_1,swap_loss_2)
        loss = loss_1+loss_2
        loss.backward() 
        optimizer.step() # update para.
        
        scheduler.step() #lr decay
        print('epoch: {} loss: {}'.format(epoch, loss))
        model.eval()
    
        C = (new_c_1 + new_c_2)/2 #average soft labels
        #Results
        predicted_labels = torch.argmax(C, dim=1).data.cpu().numpy() # min(predicted_labels)=0
        acc, nmi, ari, f1 = eva(label, predicted_labels) 
    #mem_used = torch.cuda.memory_allocated(device=args.device)
    #max_mem_a = torch.cuda.max_memory_allocated(device=args.device)
    #z=(z_igae_1+z_igae_2)/2
    #z = z.cpu()
    #z=z.detach().numpy()

    #print(f"Model: Max memory: {max_mem_a / (1024 ** 2):.2f} MB, Total memory: {mem_used/ (1024 ** 2):.2f} MB")
    #print("The total number of parameters is: " + str(count_parameters(model)) + "M(1e6).")
    return acc
  
