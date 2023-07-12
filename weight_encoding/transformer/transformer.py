import torch.nn as nn
from weight_encoding.transformer.encoder import *
from weight_encoding.transformer.decoder import *

class Transformer(nn.Module):
    
    def __init__(self,device):
        super().__init__()
        self.encoder = Encoder(device).to(device)
        self.decoder = Decoder(device).to(device)
        
    
    def forward(self, x):
        x=self.encoder(x)
        x=self.decoder(x,x)
        return x