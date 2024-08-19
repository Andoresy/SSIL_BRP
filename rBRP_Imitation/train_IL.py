import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from time import time
from datetime import datetime
import os
import numpy as np
import random
from model_LSTM import AttentionModel_LSTM
from data import Generator
import copy
from params import argparse_train_IL
def reset_SEED(SEED=2024):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def train(log_path = None, dict_file = None):
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        device = 'cuda:0'
        torch.cuda.set_device(device)
    else:
        device = 'cpu'

    n_encode_layers = dict_file["n_encode_layers"] 
    epochs = dict_file["epochs"] 
    batch = dict_file["batch"] 
    batch_verbose = dict_file["batch_verbose"] 
    max_stacks = dict_file["max_stacks"] 
    max_tiers = dict_file["max_tiers"]
    lr = dict_file["lr"]
    ff_hidden = dict_file["ff_hidden"]
    embed_dim = dict_file["embed_dim"]
    n_heads = dict_file["n_heads"]
    n_containers = max_stacks*(max_tiers-2)
    model_save_path = log_path
    log_path = log_path + f'/IL_{max_stacks}X{max_tiers-2}.txt'

    # Check if a file is already created
    
    with open(log_path, 'w') as f:
        f.write(datetime.now().strftime('%y%m%d_%H_%M'))
    with open(log_path, 'a') as f:
        f.write('\n start training \n')
        f.write(dict_file.__str__())
    
    model = AttentionModel_LSTM(device=device, n_encode_layers=n_encode_layers,n_heads= n_heads, ff_hidden=ff_hidden, embed_dim=embed_dim, max_stacks = max_stacks, max_tiers = max_tiers, n_containers = n_containers)
    model=model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    #If you do not have train_data.pt and val_data.pt file, below code should be run at least once.
    """
    data = Generator()
    t_ratio = .9
    train_size = int(len(data)*t_ratio)
    val_size = len(data) - train_size
    train_data, val_data = random_split(data, [train_size, val_size])
    dataloader = DataLoader(train_data, batch_size=batch, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch, shuffle=False)
    torch.save(dataloader, "rBRP_Imitation/train_data/train_data.pt")
    torch.save(val_dataloader, "rBRP_Imitation/train_data/val_data.pt")
    """ 
    
    dataloader = torch.load("rBRP_Imitation/train_data/train_data.pt")
    val_dataloader = torch.load("rBRP_Imitation/train_data/val_data.pt")

    def bc_loss(model, inputs, labels, t, device):
        inputs = copy.deepcopy(inputs).to(device)
        ps = torch.softmax(model(inputs), dim=1)
        labels = torch.tensor(labels).to(device).view(-1,1) % max_stacks
        loss = - torch.log(torch.gather(ps, dim=1, index=labels))
        return loss.mean()

    tt1 = time()
    t1=time()

    data = None #
    for epoch in range(epochs):
        ave_loss, ave_L = 0., 0.
        t = 0
        model.train()
        for _, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.view(len(inputs), max_stacks, max_tiers)
            t += 1
            loss = bc_loss(model,inputs,labels,t,device)
            optimizer.zero_grad()
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
            optimizer.step()

            ave_loss += loss.item()

            if t % (batch_verbose) == batch_verbose-1:
                t2 = time()
                print('Epoch %d (batch = %d): Loss: %1.3f , batch_Loss: %1.3f %dmin%dsec' % (
                    epoch, t, ave_loss / (t + 1), loss.item(), (t2 - t1) // 60, (t2 - t1) % 60))
                if True:
                    with open(log_path, 'a') as f:
                        f.write('Epoch %d (batch = %d): Loss: %1.3f , batch_Loss: %1.3f %dmin%dsec\n' % (
                    epoch, t, ave_loss / (t + 1), loss.item(), (t2 - t1) // 60, (t2 - t1) % 60))
                t1 = time()
        val_loss = 0
        val_n = 0
        model.eval()
        for _, (inputs, labels) in enumerate(val_dataloader):
            val_loss += bc_loss(model,inputs,labels,t,device)*len(inputs)
            val_n += len(inputs)
        val_loss/= val_n
        print(f"epoch {epoch} val_loss: ", val_loss.item())
        with open(log_path, 'a') as f:
            f.write(f"epoch {epoch} val_loss: {val_loss.item()} \n")
        torch.save(model.state_dict(), model_save_path + '/epoch%s.pt' % (epoch))

    tt2 = time()
    print('all time, %dmin%dsec' % (
        (tt2 - tt1) // 60, (tt2 - tt1) % 60))

if __name__ == '__main__':
    args = argparse_train_IL()
    SEED = args.seed
    reset_SEED(SEED)
    params = {"n_encode_layers": args.n_encode_layers,
                 "epochs": args.epoch,
                 "batch": args.batch,
                 "batch_verbose": args.batch_verbose,
                 "max_stacks": 5, #This is fixed for training in IL
                 "max_tiers": 7, #This is fixed for training in IL
                 "plus_tiers": 2, #This is fixed for training in IL
                 "lr": args.lr,
                 "embed_dim": args.embed_dim,
                 "ff_hidden" : args.ff_hidden_dim,
                 "n_heads" : args.n_heads}
    #Create new path
    i = 0 
    newpath = f'rBRP_Imitation/train/Exp{i}' 
    while os.path.exists(newpath):
        i = i+1
        newpath = f'rBRP_Imitation/train/Exp{i}'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    train(log_path = newpath, dict_file=params)