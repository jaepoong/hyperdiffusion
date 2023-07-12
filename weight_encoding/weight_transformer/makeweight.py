import torch
import os

weight_dir = '/nas2/lait/tjfwownd/hyperdiff/data/photoshape_weight'
weight_list = [os.path.join(weight_dir, name) for name in os.listdir(weight_dir) if os.path.exists(os.path.join(weight_dir, name, 'checkpoints', 'ngp.pth'))]

for path in weight_list:
    ckpt = torch.load(os.path.join(path,'checkpoints','ngp.pth'),map_location='cpu')
    
    z= torch.zeros((493*1024))
    n = 0 
    for k in [k for k in ckpt['model'].keys() if 'param' in k] :
        param = ckpt['model'][k]
        z[n:n + param.numel()] = param.flatten()
        n += param.numel()
    weight_path=(path.split('/'))[-1]
    torch.save(z,os.path.join('/nas2/lait/tjfwownd/hyperdiff/data/photoshape_weight/weight',(weight_path+".bin")))
    print(weight_path)