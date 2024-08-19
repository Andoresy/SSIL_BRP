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
from baseline import load_model
from data import Generator
from self_supervised_sampling import SSIL_baseline
from params import argparse_train_SSIL
import copy

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
    batch_verbose = dict_file["batch_verbose"] 
    max_stacks = dict_file["max_stacks"] 
    max_tiers = dict_file["max_tiers"]
    plus_tiers = dict_file["plus_tiers"]
    lr = dict_file["lr"]
    ff_hidden = dict_file["ff_hidden"]
    batch = dict_file["batch"]
    embed_dim = dict_file["embed_dim"]
    n_heads = dict_file["n_heads"]
    model_path = dict_file["model_path"]
    n_problems = dict_file["n_problems"]
    n_samplings = dict_file["n_samplings"]
    alpha_c = dict_file["alpha_c"]
    alpha_r = dict_file["alpha_r"]

    model_save_path = log_path
    log_path = log_path + f'/SSIL_{max_stacks}X{max_tiers-2}.txt'

    with open(log_path, 'w') as f:
        f.write(datetime.now().strftime('%y%m%d_%H_%M'))
    with open(log_path, 'a') as f:
        f.write('\n start training \n')
        f.write(dict_file.__str__())

    model = load_model(device=device, path=model_path,n_encode_layers=n_encode_layers, embed_dim=embed_dim, max_stacks = max_stacks, max_tiers = max_tiers+plus_tiers-2,
                       ff_hidden=ff_hidden, n_heads=n_heads)
    
    model=model.to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
 
    def bc_loss(model, inputs, labels, t, device):
        inputs = copy.deepcopy(inputs).to(device)
        ps = torch.softmax(model(inputs), dim=1)
        labels = torch.tensor(labels).to(device).view(-1,1) % max_stacks
        loss = - torch.log(torch.gather(ps, dim=1, index=labels))
        return loss.mean()

    tt1 = time()
    t1=time()

    baseline = SSIL_baseline(device,model, max_stacks, max_tiers, n_problems=3200, alpha_c=alpha_c, alpha_r = alpha_r)
    data = None #
    for epoch in range(epochs):
        ave_loss, ave_L = 0., 0.
        t = 0
        if epoch == 0:
            with torch.no_grad():
                ss_data = baseline.create_ss_samples(device, max_stacks=max_stacks, max_tiers =max_tiers, n_problems = n_problems, n_samplings=n_samplings)
            print('Initializing Data at epoch 0')
            with open(log_path, 'a') as f:
                f.write('Initializing Data at epoch 0\n')
            data = Generator(ss_data)
            dataloader = DataLoader(data, batch_size=batch, shuffle=True)
        elif baseline.callback(device, model, max_stacks, max_tiers, epoch=epoch - 1):
            model.eval()
            print(f'model commited for baseline at epoch{epoch}\n')
            with open(log_path, 'a') as f:
                f.write(f'model commited for baseline at epoch{epoch}\n')
            with torch.no_grad():
                ss_data = baseline.create_ss_samples(device, max_stacks=max_stacks, max_tiers =max_tiers, n_problems = n_problems, n_samplings=n_samplings)
            data = Generator(ss_data)
            dataloader = DataLoader(data, batch_size=batch, shuffle=True)
        elif baseline.callback_rollback(device, model, max_stacks, max_tiers, epoch=epoch - 1):
            model = copy.deepcopy(baseline.baseline_model)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            print(f'model rollback to baseline at epoch{epoch}\n')
            with open(log_path, 'a') as f:
                f.write(f'model rollback to baseline at epoch{epoch}\n')
            with torch.no_grad():
                ss_data = baseline.create_ss_samples(device, max_stacks=max_stacks, max_tiers =max_tiers, n_problems = n_problems, n_samplings=n_samplings)
            data = Generator(ss_data)
            dataloader = DataLoader(data, batch_size=batch, shuffle=True)
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
        
        model.eval()
        torch.save(model.state_dict(), model_save_path + '/epoch%s.pt' % (epoch))

    tt2 = time()
    print('all time, %dmin%dsec' % (
        (tt2 - tt1) // 60, (tt2 - tt1) % 60))

if __name__ == '__main__':
    args = argparse_train_SSIL()
    SEED = args.seed
    reset_SEED(SEED)
    params = {"n_encode_layers": args.n_encode_layers,
                 "epochs": args.epoch,
                 "batch": args.batch,
                 "batch_verbose": args.batch_verbose,
                 "max_stacks": args.max_stacks,
                 "max_tiers": args.max_tiers, 
                 "plus_tiers": 2, #fixed for H_max = H + 2
                 "lr": args.lr,
                 "embed_dim": args.embed_dim,
                 "ff_hidden" : args.ff_hidden_dim,
                 "n_heads" : args.n_heads,
                 "n_problems" : args.problem_num,
                 "n_samplings" : args.sampling_num,
                 "alpha_c" : args.commit_alpha,
                 "alpha_r" : args.rollback_alpha,
                 "model_path" : rf"rBRP_Imitation\train\pre_trained\{args.model_path}"
                 }
    
    #Create new path
    i = 0 
    newpath = f'rBRP_Imitation/train/Exp{i}' 
    while os.path.exists(newpath):
        i = i+1
        newpath = f'rBRP_Imitation/train/Exp{i}'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    train(log_path = newpath, dict_file=params)