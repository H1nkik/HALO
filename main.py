# -*- coding: utf-8 -*-
import opt
from utils import *
from GAE import *
from train import Train_gae
import time_manager
import warnings

warnings.filterwarnings('ignore')

# 自动检测设备（如果未指定或为auto）
if not hasattr(opt.args, 'device') or opt.args.device == 'auto':
    opt.args.device = get_device()

setup()

# 加载数据集
print("\n" + "=" * 60)
print("加载数据集")
print("=" * 60)
x, y, adj, _, _ = load_graph_data(opt.args.dataset, show_details=True)
x = ensure_float32(x, opt.args.device)

model = GAEC(
    gae_n_enc_1=opt.args.gae_n_enc_1,
    gae_n_enc_2=opt.args.gae_n_enc_2,
    gae_n_enc_3=opt.args.gae_n_enc_3,
    n_input=opt.args.n_input,
    model_type=opt.args.model_type,
    sage_aggr=opt.args.sage_aggr,
    sage_dropout=opt.args.sage_dropout
).to(opt.args.device)

print("\n开始训练...")
print("-" * 60)

timer = time_manager.MyTime()
timer.start()
acc, metrics = Train_gae(model, x, adj, y)
seconds, minutes = timer.stop()

print("-" * 60)
print("训练完成!")
print(f"最终结果: ACC={acc:.4f}, NMI={metrics['nmi']:.4f}, ARI={metrics['ari']:.4f}, F1={metrics['f1']:.4f}")
print(f"耗时: {seconds}s 或 {minutes}m")
print("=" * 60)
