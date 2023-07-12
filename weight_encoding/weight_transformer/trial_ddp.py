from dataset import WeightDataset,EvalDataset
from torch.utils.data import DataLoader,Dataset
from model import SampleTransformer
from vaemodel import VAE_transformer
from util import *
import os
import sys
from os.path import dirname,abspath
sys.path.append(dirname(dirname(dirname(abspath(__file__)))))

from nerf.network_tcnn import NeRFNetwork
import numpy as np
from nerf.utils import *
from nerf.provider import *
from tqdm import tqdm
import argparse

import torch
from torch import optim, nn
from torch.cuda.amp import autocast
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP




def evaluate_nerf(eval_loader,device,epoch,nerf_model,net,data):
    net.eval()
    weight,data_path=next(iter(eval_loader))
    weight=weight.to(device)

    param=[p.numel() for p in nerf_model.parameters() if p.requires_grad]
    index1=param[0]
    index2=index1+param[1]
    
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            pred = net(weight)[0]
        example_ckpt['model']['encoder.params']=pred[:index1]
        example_ckpt['model']['sigma_net.params']=pred[index1:index2]
        example_ckpt['model']['color_net.params']=pred[index2:]
        
        nerf_model.load_state_dict(example_ckpt['model'])
        loss = nn.L1Loss()(weight[0],pred)
        log(f"eval_loss at {epoch} epoch : loss = {loss.item()} : eval_root= {data_path}")
        with autocast():
            result = nerf_model.render(data['rays_o'],data['rays_d'],staged=True,perturb=False,bg_color=1)
        
    pred=result['image'].reshape(-1,256,256,3)
    pred=pred.detach().cpu().numpy()
    pred=(pred*255).astype(np.uint8)
    imageio.mimwrite(os.path.join("/nas2/lait/tjfwownd/hyperdiff/weight_encoding/weight_transformer/result/nerf_sample/",f'ddp_sample_epoch{epoch}.mp4'),pred, fps=8, quality=8, macro_block_size=1)   
     
example_ckpt = torch.load("/nas2/lait/tjfwownd/hyperdiff/data/photoshape_weight/shape09096_rank01/checkpoints/ngp.pth",map_location='cpu')

def avg(data) :
    return sum(data) / len(data)

log_path = "result/log_data_all_ddp.txt"
log_ptr = open(log_path,'a+')

def log(*args, **kwargs):
    print(*args, file=log_ptr)
    log_ptr.flush() # write immediately to file

def setup(rank,world_size,port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group('nccl',rank=rank,world_size=world_size)


set_seed(42)

def train(rank,world_size,opt):
    
    setup(rank,world_size,opt.port)
    device = torch.device(rank)
    
    nerf_opt = get_opt_for_nerf()
    # data define
    weight_dir = '/nas2/lait/tjfwownd/hyperdiff/data/photoshape_weight/weight'
    weight_dataset = WeightDataset(device=device,weight_dir=weight_dir)
    eval_dataset = EvalDataset(device=device,weight_dir=weight_dir)

    sampler = torch.utils.data.distributed.DistributedSampler(weight_dataset,
                                                              num_replicas=world_size,
                                                              rank=rank)
    loader = DataLoader(
        dataset=weight_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=12,
        sampler = sampler
    )
    eval_loader= DataLoader(dataset=eval_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=12)


    # network define
    net = SampleTransformer(device).to(device)
    net = DDP(net, device_ids=[rank])
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

    focal=245/(2*np.tan(np.radians(60)/2))
    intrinsics=np.array([focal,focal,256//2,256//2])
    poses=line_poses(40,device,radius=0.5)
    data=get_rays(poses,intrinsics,256,256)

    # training option
    optimizer = optim.AdamW(net.parameters(), lr=1e-5)
    criterion = nn.L1Loss()
    epochs = 10000
    best = 10000
    eval_interval = 150

    schedule = "Cosine_warmup"
    if schedule == 'Cosine_warmup':
        lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup= 100, max_iters=epochs)
    else:
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer,lambda step: 1-step/epochs)

    q=0
    scaler = torch.cuda.amp.GradScaler()
    
    master_process = rank==0
    for epoch in range(epochs) :
        net.train()

        t_loss = list()

        #pbar = tqdm(loader)
        with tqdm(loader, disable = not master_process) as pbar :
            for weight in loader :
                weight = weight.to(device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    pred = net(weight)
                    loss = criterion(weight, pred)
                scaler.scale(loss).backward() 
                scaler.step(optimizer)
                scaler.update()
                t_loss.append(loss.item())
                if rank==0:
                    pbar.set_description('[ep %d/%d] loss : %.4f (avg: %.4f)' % (epoch + 1, epochs, loss.item(), avg(t_loss)))
                    pbar.update()
        lr_scheduler.step()
        average = avg(t_loss)
        
        if rank==0 and best > average :
            q+=1
            log(f"loss:{average}")
            if q%10==0:
                best = average
                if epoch >=10:
                    torch.save(pred, 'result/best_data_all_ddp.bin')
                    torch.save(net.state_dict(), 'result/best_data_all_ddp.pth')
        
        if rank ==0 and epoch % eval_interval == 0:
            evaluate_nerf(eval_loader,device,epoch,nerf_model,net,data)
            
            

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs",type=int, default= 10000, help = " number_of_epoch in training, per entire data")
    parser.add_argument("--port", type=str, default='12355')
    
    opt = parser.parse_args()
    print(opt)
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    mp.spawn(train, args=(num_gpus,opt), nprocs=num_gpus, join=True)
