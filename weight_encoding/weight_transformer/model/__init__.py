from .model import *

def load_model(args,device):
    if args.model =="Basic_Transformer":
        model = Basic_Transformer(device,args.num_layer,args.latent_dim)
    elif args.model == "Latent_injection_Transformer":
        model = Latent_injection_Transformer(device,args.num_layer,args.latent_dim)
    else:
        raise NotImplementedError("args.model must be in model_zoo")
    return model