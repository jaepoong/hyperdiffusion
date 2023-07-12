from dataset import WeightDataset
from torch.utils.data import DataLoader
from vaemodel import VAE_transformer
from tqdm import tqdm
from torch import optim, nn
import torch

weight_dir = '/nas2/lait/tjfwownd/hyperdiff/data/photoshape_weight/weight'

weight_dataset = WeightDataset(weight_dir)

loader = DataLoader(
    dataset=weight_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=12
)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#net = SampleTransformer()
net = VAE_transformer(device=device).to(device)


optimizer = optim.AdamW(net.parameters(), lr=1e-5)
criterion = nn.L1Loss()

net.to(device)

epochs = 100000
best = 10000
beta = 1 
def avg(data) :
    return sum(data) / len(data)

log_path = "log_vae.txt"

log_ptr = open(log_path,'a+')
def log(*args, **kwargs):
    print(*args, file=log_ptr)
    log_ptr.flush() # write immediately to file

for epoch in range(epochs) :
    net.train()

    t_loss = list()

    pbar = tqdm(loader)
    for weight in pbar :
        weight = weight.to(device)
        optimizer.zero_grad()
        pred,mu,var = net(weight)
        # loss
        reconstruction_loss = criterion(pred,weight)
        kld_loss = torch.mean(-0.5*torch.sum(1 + var - mu**2 -var.exp(),dim=[1,2]),dim=0)
        loss = reconstruction_loss+beta*kld_loss # Based on Beta-VAE!! Beta로 KL term reg. + LDM used 10^-6 KL reg term for powerfull reconstruction quality.
        loss.backward()
        optimizer.step()

        t_loss.append(loss.item())
        
        pbar.set_description('[ep %d/%d] loss : %.4f, reconstruction_loss : %.4f, kld_loss : %.4f (avg: %.4f) ' % (epoch + 1, epochs, loss.item(), reconstruction_loss.item(), kld_loss.item(), avg(t_loss)))
    average = avg(t_loss)
    
    if best > average :
        best = average
        torch.save(pred, 'best_vae.bin')
        torch.save(net.state_dict(), 'best_vae.pth')
        log(f"loss:{average}")
        