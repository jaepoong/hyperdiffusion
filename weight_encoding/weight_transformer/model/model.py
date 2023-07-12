from diffusers.models.attention import BasicTransformerBlock
from torch import nn
import torch

class Basic_Transformer(nn.Module) :
    def __init__(self,device,num_layer=168,latent_dim=128) :
        super().__init__()
        self.device=device
        self.pos_encoding = PositionalEncoding(1024,493,self.device)
        self.blocks = nn.ModuleList([BasicTransformerBlock(1024, 16, 64, dropout=0.2) for _ in range(num_layer)])
        self.blocks.append(nn.Linear(1024,latent_dim))
        self.blocks.append(nn.Linear(latent_dim,1024))
        for i in range(num_layer):
            self.blocks.append(BasicTransformerBlock(1024,16,64,dropout=0.2))

    def forward(self, x) :
        x = x.view(-1, 493, 1024).to(self.device)
        x = self.pos_encoding(x)
        for block in self.blocks :
            x = block(x)
        return x.flatten(start_dim=1,end_dim=2)
        
class Latent_injection_Transformer(nn.Module) :
    def __init__(self,device,num_layer=18,latent_dim=128) :
        super().__init__()
        self.device=device
        self.pos_encoding = PositionalEncoding(1024,493,device)
        self.encoder = nn.ModuleList([BasicTransformerBlock(1024, 16, 64, dropout=0.2) for _ in range(num_layer)])
        self.latent_en= nn.Linear(1024,latent_dim)
        self.latent_de= nn.Linear(latent_dim,1024)
        self.decoder = nn.ModuleList([BasicTransformerBlock(1024, 16, 64, dropout=0.2) for _ in range(num_layer)])
    def forward(self, x) :
        x = x.view(-1, 493, 1024).to(self.device)
        x = self.pos_encoding(x)
        for block in self.encoder :
            x = block(x)
        latent=self.latent_en(x)
        
        latent=self.latent_de(latent)
        x = latent
        for block in self.decoder[:-1]:
            x= block(x,encoder_hidden_states=latent)
        x = self.decoder[-1](x)

        return x.flatten(start_dim=1,end_dim=2)
"""
class Funnel_Transformer(nn.Module):
    def __init__(self,d_model,max_len,d):
        pass
    
"""
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding,self).__init__()
        self.device=device
        self.encoding = torch.zeros(max_len, d_model).to(device)
        self.encoding.requires_grad = False
        
        pos = torch.arange(0,max_len).to(device)
        pos = pos.float().unsqueeze(dim=1)
        
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        
        self.encoding[:,0::2] = torch.sin(pos/(10000 **(_2i / d_model)))
        self.encoding[:,1::2] = torch.cos(pos/(10000**(_2i / d_model)))
        
    
    def forward(self, x):
        batch,seq_len,d = x.size()
        
        return self.encoding[:seq_len, :].unsqueeze(0) + x