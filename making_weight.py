import torch
from nerf.provider import NeRFDataset
import argparse
from nerf.utils import *
from nerf.network import NeRFNetwork
from encoding import *
print('ji')

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='./data/photoshapes/shape09096_rank00',type=str)
parser.add_argument('-O', action='store_false', help="equals --fp16 --cuda_ray --preload")
parser.add_argument('--test', action='store_true', help="test mode")
parser.add_argument('--workspace', type=str, default='workspace')
parser.add_argument('--seed', type=int, default=0)
### training options
parser.add_argument('--iters', type=int, default=15000, help="training iters")
parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
parser.add_argument('--ckpt', type=str, default='best')
parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
parser.add_argument('--num_steps', type=int, default=512, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")
parser.add_argument('--patch_size', type=int, default=1, help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")
### network backbone options
parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")
### dataset options
parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")
# (the default value is for the fox dataset)
parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")
parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
parser.add_argument('--dt_gamma', type=float, default=1/128, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
parser.add_argument('--min_near', type=float, default=0.2, help="minimum near distance for camera")
parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
parser.add_argument('--bg_radius', type=float, default=-1, help="if positive, use a background model at sphere(bg_radius)")
### GUI options
parser.add_argument('--gui', action='store_true', help="start a GUI")
parser.add_argument('--W', type=int, default=1920, help="GUI width")
parser.add_argument('--H', type=int, default=1080, help="GUI height")
parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")
### experimental
parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")
parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
parser.add_argument('--rand_pose', type=int, default=-1, help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")



opt = parser.parse_args(args=[])



if opt.O:
    opt.fp16 = True
    opt.cuda_ray = False # if use this, model make density bitfield. So making field would hard.
    opt.preload = True

if opt.patch_size > 1:
    opt.error_map = False # do not use error_map if use patch-based training
    # assert opt.patch_size > 16, "patch_size should > 16 to run LPIPS loss."
    assert opt.num_rays % (opt.patch_size ** 2) == 0, "patch_size ** 2 should be dividable by num_rays."

opt.cuda_ray=False 

import os
from nerf.network_tcnn import NeRFNetwork

data_dir = "./data/omni3d/OpenXD-OmniObject3D-New/raw/blender_renders"
paths=sorted(os.listdir(data_dir))[2:28]
seed_everything(opt.seed)

i=0
for path in paths:
    print(path)
    data_path=os.path.join(data_dir,path,'render')
    work_space=os.path.join('./result',path)
    opt.path=data_path
    
    opt.workspace=work_space
    # model define

    model = NeRFNetwork(
        encoding="hashgrid",
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
    )

    # loss define    
    criterion = torch.nn.MSELoss(reduction='none')
    
    # device define
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # train_loader define
        #train_loader = NeRFDataset(opt, device=device, type='trainval').dataloader()
    train_loader = NeRFDataset(opt, device=device, type='train').dataloader() # for omni3d

    
    # optimizer define
    optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

    # scheduler define
    scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

    # metric define
    metrics = [PSNRMeter(), LPIPSMeter(device=device)]

    # trainer define
    trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, 
                  optimizer=optimizer, criterion=criterion, ema_decay=0.95, 
                  fp16=opt.fp16, lr_scheduler=scheduler, scheduler_update_every_step=True, 
                  metrics=metrics, use_checkpoint=opt.ckpt, eval_interval=50)

    # valid_loader define
    valid_loader = NeRFDataset(opt, device=device, type='val', downscale=1).dataloader()
    
    max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)

    trainer.train(train_loader, valid_loader, max_epoch)

    #test_loader = NeRFDataset(opt, device=device, type='test').dataloader()
