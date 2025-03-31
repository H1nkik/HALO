# -*- coding: utf-8 -*-
import opt 
from utils import * 
from GAE import *
from train import Train_gae
import time_manager
import warnings

warnings.filterwarnings('ignore')

setup()

x, y, adj,_,_ = load_graph_data(opt.args.dataset, show_details=True) 
x = torch.from_numpy(x).to(opt.args.device).float() 

model = GAEC(
        gae_n_enc_1=opt.args.gae_n_enc_1,
        gae_n_enc_2=opt.args.gae_n_enc_2,
        gae_n_enc_3=opt.args.gae_n_enc_3,
        n_input=opt.args.n_input
    ).to(opt.args.device)

timer = time_manager.MyTime()
timer.start()
Train_gae(model,x,adj, y)
seconds, minutes = timer.stop()
print("Time consuming: {}s or {}m".format(seconds, minutes))