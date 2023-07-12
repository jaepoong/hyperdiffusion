from dataset import WeightDataset,EvalDataset
from torch.utils.data import DataLoader,Dataset
from vaemodel import VAE_transformer
from torch import optim, nn
from util import *
import torch
import os
import sys
from os.path import dirname,abspath
sys.path.append(dirname(dirname(dirname(abspath(__file__)))))


from nerf.network_tcnn import NeRFNetwork
from torch.cuda.amp import autocast
import numpy as np
from nerf.utils import *
from nerf.provider import *
from tqdm import tqdm
import time
from model import load_model
import argparse

def evaluate_nerf(eval_loader,device,epoch,args):
    net.eval()
    weight,data_path=next(iter(eval_loader))
    weight=weight.to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            pred = net(weight)[0]
        example_ckpt['model']['encoder.params']=pred[:index1]
        example_ckpt['model']['sigma_net.params']=pred[index1:index2]
        example_ckpt['model']['color_net.params']=pred[index2:]
        
        nerf_model.load_state_dict(example_ckpt['model'])
        loss = criterion(weight[0],pred)
        log(f"eval_loss at {epoch} epoch : loss = {loss.item()} : eval_root= {data_path}")
        with autocast():
            result = nerf_model.render(data['rays_o'],data['rays_d'],staged=True,perturb=False,bg_color=1)
        
    pred=result['image'].reshape(-1,256,256,3)
    pred=pred.detach().cpu().numpy()
    pred=(pred*255).astype(np.uint8)
    if not os.path.isdir(os.path.join(args.work_space,args.work_name,"nerf_sample")):
        os.makedirs(os.path.join(args.work_space,args.work_name,"nerf_sample"), exist_ok=True)
    imageio.mimwrite(os.path.join(args.work_space,args.work_name,"nerf_sample", f'128_sample_epoch {epoch} .mp4'),pred, fps=8, quality=8, macro_block_size=1)    

set_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", help = "If load ckpt, give path about the pth")
parser.add_argument("--work_space",default = "/nas2/lait/tjfwownd/hyperdiff/weight_encoding/weight_transformer/work_space", help = "work_space_name. after training all of result is saved to this directory")
parser.add_argument("--work_name",default = "trial", help = "this argument for trial name.")
parser.add_argument("--weight_dir", default = "/nas2/lait/tjfwownd/hyperdiff/data/photoshape_weight/weight", help = " weight dataset-dir")
parser.add_argument("--scheduler", default = "Cosine_warmup_restarts")
parser.add_argument("--num_layer", default = 18 , help = " transformer layer number",type = int)
parser.add_argument("--latent_dim", default= 128 , help =  " tranformer latent dim", type = int)
parser.add_argument("--model", default = "Latent_injection_Transformer", type = str)
parser.add_argument("--optimizer", default="Adam",choices=["Adam","AdamW"])
args = parser.parse_args()

if not os.path.isdir(os.path.join(args.work_space,args.work_name)):
    os.makedirs(os.path.join(args.work_space,args.work_name), exist_ok=True)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nerf_opt = get_opt_for_nerf()
# data define
weight_dir = args.weight_dir
weight_dataset = WeightDataset(device=device,weight_dir=weight_dir)
eval_dataset = EvalDataset(device=device,weight_dir=weight_dir)

loader = DataLoader(
    dataset=weight_dataset,
    batch_size=3,
    shuffle=True,
    num_workers=12,
    pin_memory=True
)
eval_loader= DataLoader(dataset=eval_dataset,
                        batch_size=1,
                        shuffle=True,
                        num_workers=12,
                        pin_memory=True
                        )


# network define
net = load_model(args,device).to(device)
#net = SampleTransformer(device,num_layer=args.num_layer, latent_dim=args.latent_dim).to(device)
nerf_model = NeRFNetwork(
        encoding="hashgrid",
        bound=nerf_opt.bound,
        cuda_ray=nerf_opt.cuda_ray,
        density_scale=1,
        min_near=nerf_opt.min_near,
        density_thresh=nerf_opt.density_thresh,
        bg_radius=nerf_opt.bg_radius,
    ).to(device) # for evaluation_ predefine nerf_model
    
# evaluation
example_ckpt = torch.load("/nas2/lait/tjfwownd/hyperdiff/data/photoshape_weight/shape09096_rank01/checkpoints/ngp.pth",map_location='cpu')
param=[p.numel() for p in nerf_model.parameters() if p.requires_grad]
index1=param[0]
index2=index1+param[1]
index3=index2+param[2]

focal=245/(2*np.tan(np.radians(60)/2))
intrinsics=np.array([focal,focal,256//2,256//2])
poses=line_poses(40,device,radius=0.5)
data=get_rays(poses,intrinsics,256,256)

# training option
criterion = nn.MSELoss()
epochs = 10000
best = 10000
eval_interval = 150

# optimizer
if args.optimizer == "Adam":
    optimizer = optim.Adam(net.parameters(),lr=1e-5)
elif args.optimizer =="AdamW":
    optimizer = optim.AdamW(net.parameters(),lr=1e-5)
else:
    raise NotImplementedError("Optimizer is not defined")

if args.scheduler == 'Cosine_warmup':
    lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup= 100, max_iters=epochs)
elif args.scheduler == 'Cosine_warmup_restarts':
    for g in optimizer.param_groups:
        g['lr'] = 1e-7
    lr_scheduler = CosineAnnealingWarmUpRestarts(optimizer,T_0=1000,T_mult=1,eta_max=1e-4,T_up=100,gamma=0.7)
else:
    optimizer = optim.AdamW(net.parameters(), lr=1e-5)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer,lambda step: 1-step/epochs)

scaler = torch.cuda.amp.GradScaler()

curr_epoch = 0

# Code for checkpoint
if args.ckpt_dir:
    checkpoint = torch.load(args.ckpt_dir)
    net.load_state_dict(checkpoint['net'])
    scaler.load_state_dict(checkpoint['scaler'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    curr_epcoh = checkpoint['curr_epoch']
    best = checkpoint['best']
    

def avg(data) :
    return sum(data) / len(data)

log_path = os.path.join(args.work_space, args.work_name,"log.txt")
log_ptr = open(log_path,'a+')

def log(*args, **kwargs):
    print(*args, file=log_ptr)
    log_ptr.flush() # write immediately to file
log(vars(args))

q=0
# --------
# Training
# --------
for epoch in range(curr_epoch,epochs) :
    curr_epoch +=1
    net.train()

    t_loss = list()

    pbar = tqdm(loader)
    for weight in pbar :
        weight = weight.to(device)
        optimizer.zero_grad() 
        
        with torch.cuda.amp.autocast():
            pred = net(weight)
            loss = criterion(weight, pred)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        t_loss.append(loss.item())
        
        pbar.set_description('[ep %d/%d] loss : %.4f (avg: %.4f)' % (epoch + 1, epochs, loss.item(), avg(t_loss)))
    lr_scheduler.step()
    average = avg(t_loss)
    
    if best > average :
        q+=1
        log(f"loss:{average}")
        if q%10==0:
            time.sleep(5)
            best = average
            if epoch >=10:
                torch.save(pred, os.path.join(args.work_space,args.work_name,"best.bin"))
                # Code for checkpoint
                torch.save({
                    'net' : net.state_dict(),
                    'curr_epoch' : curr_epoch,
                    'best' : best,
                    'optimizer' : optimizer.state_dict(),
                    'scaler' : scaler.state_dict()
                    }, os.path.join(args.work_space,args.work_name,f"checkpoint.pth"))

    
    if epoch % eval_interval == 0:
        evaluate_nerf(eval_loader,device,epoch,args)



            
            
    