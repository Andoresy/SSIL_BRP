import torch
import torch.nn as nn
from model_LSTM import AttentionModel_LSTM

def load_model(device,path,embed_dim,max_stacks,max_tiers,n_encode_layers,ff_hidden, n_heads):
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    # https://pytorch.org/docs/master/generated/torch.load.html
    model_loaded = AttentionModel_LSTM(device=device, embed_dim=embed_dim, n_encode_layers=n_encode_layers,n_heads=n_heads,
                                        max_stacks=max_stacks, max_tiers=max_tiers, ff_hidden=ff_hidden)
    if torch.cuda.is_available():
        model_loaded.load_state_dict(torch.load(path,map_location={'cuda:0' : device ,'cuda:1': device , 'cuda:2' :device ,
                                                                   'cuda:3' : device ,'cuda:4': device , 'cuda:5' :device}))
    else:
        model_loaded.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    return model_loaded

