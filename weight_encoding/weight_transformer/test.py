from dataset import WeightDataset
from torch.utils.data import DataLoader
from model import SampleTransformer
import torch

import os 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

weight_dir = '/nas2/lait/tjfwownd/hyperdiff/data/photoshape_weight/weight'
weight_dataset = WeightDataset(weight_dir)
loader = DataLoader(dataset = weight_dataset,
                   batch_size = 5,
                   shuffle = False,
                   num_workers=12
                    )

net = SampleTransformer()
ckpt = torch.load('result/best_data100.pth',map_location=device)

net.load_state_dict(ckpt)

for weight in loader:
    weight.to(device)
    result = net(weight)
    
    torch.save(result, "result/test_data100.bin")
    break
