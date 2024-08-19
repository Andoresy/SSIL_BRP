from params import *
from baseline import load_model
from data import data_from_caserta
import torch
import numpy as np
from Env import Env
from tqdm import tqdm
import copy
from sampler import TopKSampler, New_Sampler
import time
import random

def reset_seed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
def test_greedy_all(device, model):
    HWS = [(3,3),(3,4),(3,5),(3,6),(3,7),(3,8),(4,4),(4,5),(4,6),(4,7),(5,4),(5,5),(5,6),(5,7), (5,8), (5,9), (5,10),(6,6), (6,10), (10,6), (10,10)]
    for H,W in HWS:
        test_greedy(device, model, H, W)

def test_greedy(device, model, H=5, W=5):
    data_caserta = data_from_caserta(H,W).to(device)
    model.eval()
    t = []
    Length = torch.zeros(len(data_caserta)).to(device)
    env = Env(device = device, x = data_caserta)
    env.clear()
    trajectory = [copy.deepcopy(env.x)]
    selecter = TopKSampler() 
    t1 = time.time()
    for _ in range(200):
        if env.all_empty():
            break
        logits = model(env.x)
        output = torch.softmax(logits, dim=1)
        is_cycle = True
        while is_cycle:
            next_action = selecter(output)
            source_node, dest_node = next_action//W, next_action%W
            actions = torch.cat((source_node,dest_node), 1)
            temp_Length = Length + (1.0 - env.empty.type(torch.float64))
            prev_x = env.x.clone()
            env.step(actions)
            is_cycle = False
            for t in trajectory:
                for i in range(len(t)):
                    if env.empty[i]:
                        continue
                    if torch.sum((t[i]!=env.x[i]).long())== 0:
                        logits[i][next_action[i]] -= 1e9
                        output = torch.softmax(logits, dim=1)
                        is_cycle = True
            if is_cycle:
                env = Env(device, prev_x)
                env.clear()
        trajectory.append(copy.deepcopy(env.x))
        if len(trajectory) > 60:
            trajectory.pop(0) #Use this when time takes too long
            pass
        Length = temp_Length
    t2 = time.time()
    print('-------------------')
    #print(Length)
    print(f"{H}X{W} Greedy Mean Locations:",np.array(Length.cpu()).mean())  # cost: (batch)
    print(f"time: {(t2-t1)/40:.2f}")

def test_sampling_all(device, model, SEED, sampling_batch, total_sampling, is_ESS, T = [1]*21 ):
    HWS = [(3,3),(3,4),(3,5),(3,6),(3,7),(3,8),(4,4),(4,5),(4,6),(4,7),(5,4),(5,5),(5,6),(5,7), (5,8), (5,9), (5,10),(6,6), (6,10), (10,6), (10,10)]
    model.eval()
    for c in range(len(HWS)):
        H,W = HWS[c]
        t = T[c]
        test_sampling(device, model, SEED, sampling_batch, total_sampling, is_ESS, H,W,t)

def test_sampling(device, model, SEED, sampling_batch, total_sampling, is_ESS, H=5, W=5, temp=1):
    t = temp
    data_caserta = data_from_caserta(H,W).to(device)
    selecter = New_Sampler(T=t)
    total_length = []
    t1 = time.time()
    reset_seed(SEED)
    for i in tqdm(range(len(data_caserta))):
        temp_length = []
        for _ in range(total_sampling//sampling_batch):
            Length = torch.zeros(sampling_batch).to(device)
            env = Env(device = device, x = data_caserta[i:i+1].repeat(sampling_batch,1,1))
            env.clear()
            for step in range(500):
                if is_ESS:
                    if env.empty.any().item():
                        break
                else:
                    if env.all_empty():
                        break
                output = model(env.x)
                next_action = selecter(output)
                source_node, dest_node = next_action//W, next_action%W
                actions = torch.cat((source_node,dest_node), 1)
                Length += (1.0 - env.empty.type(torch.float64))
                env.step(actions)
            temp_length.append(int(torch.min(Length[0]).item()))
            
        total_length.append(min(temp_length))
    t2 = time.time()
    print('----------')
    print(f"{H}X{W} Sampling {total_sampling} Times  with Temp = {selecter.T} Mean Locations:",sum(total_length)/len(total_length))  # cost: (batch)
    print(f"time: {(t2-t1)/40:.3f}")


if __name__ == '__main__':
    args = argparse_test()

    if torch.cuda.is_available():
        device = 'cuda:0'
        torch.cuda.set_device(device)
    else:
        device = 'cpu'

    model_path = f"uBRP_Imitation/train/pre_trained/{args.model_path}"
    model = load_model(device=device, path=model_path,n_encode_layers=args.n_encode_layers,
                        embed_dim=args.embed_dim, max_stacks=args.max_stacks, max_tiers=args.max_tiers, n_heads = args.n_heads, ff_hidden=args.ff_hidden_dim).to(device)

    decode_type = args.decode_type
    #If ALL:
    if args.test_all:
        if decode_type == 'greedy':
            test_greedy_all(device, model)
            pass
        else:
            is_ESS = decode_type == 'ESS'
            Ts = [1]*21 #Default
            Ts = [1,3,2,5,2,2,4,2,3,4,5,4,4,4,4,3,3,3,2,2,2] #Use this in uBRP_SSIL model for reproductivity
            test_sampling_all(device, model, args.model_path, args.seed, args.batch, args.sampling_num, is_ESS, Ts)
    else:    
        if decode_type == 'greedy':
            test_greedy(device, model, args.max_tiers-2, args.max_stacks)
        else:
            is_ESS = decode_type == 'ESS'
            test_sampling(device, model, args.seed, args.batch, args.sampling_num, is_ESS, args.temp)
