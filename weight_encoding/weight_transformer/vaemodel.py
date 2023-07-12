from diffusers.models.attention import BasicTransformerBlock
from torch import nn
import torch

class VAE_transformer(nn.Module) :
    def __init__(self, device,latent_dim = 64) :
        super(VAE_transformer,self).__init__()
        self.latent_dim=latent_dim
        self.device= device
        self.encoder = []

        for i in range(14):
            self.encoder +=[BasicTransformerBlock(1024,16,64,dropout=0.2)]
        self.encoder = nn.Sequential(*self.encoder)
        
        self.fc_mu = nn.Linear(1024,self.latent_dim)
        self.fc_var = nn.Linear(1024,self.latent_dim)
        
        self.decoder = []
        self.decoder += [nn.Linear(self.latent_dim,1024)]
        for i in range(14):
            self.decoder +=[BasicTransformerBlock(1024,16,64,dropout=0.2)]
        self.decoder = nn.Sequential(*self.decoder)
    
    def forward(self, input):
        input = input.view(-1,493,1024)
        mu,var = self.encode(input)
        z = self.reparameterize(mu,var)
        return [self.decode(z).flatten(start_dim=1,end_dim=2), mu, var]
    
    def encode(self, input):
        result = self.encoder(input)
        
        mu = self.fc_mu(result)
        var = self.fc_var(result)
        
        return mu,var
    
    def decode(self, z):
        
        result = self.decoder(z)
        return result

    def reparameterize(self, mu, var):
        std = torch.exp(0.5*var)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def sample(self, num_sample):
        
        z = torch.randn(num_sample, self.latent_dim).to(self.device)
        
        samples = self.decode(z)
        return samples

        
    
