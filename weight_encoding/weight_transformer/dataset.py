import os
import torch
from torch.utils.data import Dataset
#from transformers import CLIPTextModel, CLIPTokenizer
import glob

class WeightDataset(Dataset) :
    def __init__(self,device, weight_dir='/nas2/lait/tjfwownd/hyperdiff/data/photoshape_weight/weight') :
        self.weight_list = sorted(glob.glob(os.path.join(weight_dir,"*.bin")))[:580]
        self.device = device
    def __len__(self) :
        return len(self.weight_list)
    
    def __getitem__(self, idx) :

        ckpt = torch.load(self.weight_list[idx], map_location = 'cpu')
        return ckpt

class EvalDataset(Dataset) :
    def __init__(self,device, weight_dir='/nas2/lait/tjfwownd/hyperdiff/data/photoshape_weight/weight') :
        self.weight_list = sorted(glob.glob(os.path.join(weight_dir,"*.bin")))[580:]
        self.device = device
    def __len__(self) :
        return len(self.weight_list)
    
    def __getitem__(self, idx) :

        ckpt = torch.load(self.weight_list[idx], map_location = 'cpu')
        return ckpt,self.weight_list[idx]

class WeightDataset_Test(Dataset) :
    def __init__(self, weight_dir='/nas2/lait/tjfwownd/hyperdiff/data/photoshape_weight/weight') :
        self.weight_list = sorted(glob.glob(os.path.join(weight_dir,"*.bin")))[:580]

        #self.tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
        #self.text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder")

        #self.ckpt = torch.load(self.weight_list[0], map_location = 'cpu')
        #self.ckpt1 = torch.load(self.weight_list[100], map_location= 'cpu')
        #self.ckpt2 = torch.load(self.weight_list[200], map_location = 'cpu')
        #print(self.weight_list[0])
        #print(self.weight_list[100])
        #print(self.weight_list[200])
        print(len(self.weight_list))
    def __len__(self) :
        return len(self.weight_list)
    
    def __getitem__(self, idx) :
       
        #if idx%3==0:
        #    ckpt = self.ckpt
        #elif idx%3==1:
        #    ckpt = self.ckpt1
        #elif idx%3==2:
        #    ckpt=self.ckpt2
        ckpt = torch.load(self.weight_list[idx], map_location = 'cpu')
        return ckpt