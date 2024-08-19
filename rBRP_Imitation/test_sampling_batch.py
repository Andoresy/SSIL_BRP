from baseline import load_model
from data import data_from_caserta, data_from_caserta_for_greedy,data_from_big_instances
import torch
from Env_V3 import Env
import gc
from sampler import TopKSampler,CategoricalSampler, New_Sampler
from tqdm import tqdm
import time
import numpy as np
import random
#84, 88!
if __name__ == '__main__':
    device = 'cuda:0'
    #device = 'cpu'
    HWS = [(3,3),(3,4),(3,5),(3,6),(3,7),(3,8),(4,4),(4,5),(4,6),(4,7),(5,4),(5,5),(5,6),(5,7), (5,8), (5,9), (5,10),(6,6), (6,10), (10,6), (10,10)]
    #HWS = [(3,3), (3,4), (3,5), (3,6),(4,4),(5,6), (5,7), (3,8)]
    #HWS = [(5,9), (5,10),(6,6), (6,10), (10,6), (10,10)]
    #HWS = [(6,10),(10,6),(10,10)]
    #HWS=[(6,6)]
    TS = [1,3,2,5,2,2,4,2,3,4,5,4,4,4,4,3,3,3,2,2,2]
    for i,(H,W) in enumerate(HWS):
        SEED = 2024
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        H_plus = 2
        N = H*W
        Exp_num= 5
        epoch_num = 23
        embed_dim = 128
        data_caserta = data_from_caserta(f'data{H}-{W}-.*', H_plus).to(device)
        #data_greedy = data_from_caserta_for_greedy(f'data{H}-{W}-.*', H_plus).to(device)
        #data_caserta = data_from_big_instances(f'.*20x10.*', H_plus).to(device)
        path = f"train/pre_trained/rBRP_IL.pt"
        #path = f"../train/Exp{0}/uBRP_4x4~5x6_best.pt"
        model = load_model(device=device, path=path,n_encode_layers=3, embed_dim=embed_dim, n_containers=N, max_stacks=W, max_tiers=H+H_plus, is_Test = True).to(device)
        model.eval()
        for name, param in model.named_parameters():
            if 'resweight' in name or True :
                #print(f"Name: {name}")
                #print(f"Param: {param}")
                pass
        return_pi = False
        t = []
        
        #Greedy
        
        #T = TS[i]
        T=1
        selecter = New_Sampler(T=T)
        total_sampling = 2560
        sampling_batch = 2560
        total_length = []
        for i in tqdm(range(len(data_caserta))):
            temp_length = []
            for _ in range(total_sampling//sampling_batch):
                Length = torch.zeros(sampling_batch).to(device)
                env = Env(device = device, x = data_caserta[i:i+1].repeat(sampling_batch,1,1))
                env.clear()
                for step in range(500):
                    if torch.sum(env.empty.long()) > 0:
                        break
                    logits = model(env.x, return_pi=return_pi)
                    next_action = selecter(logits).view(-1,1)
                    Length += (1.0 - env.empty.type(torch.float64))
                    env._get_step(next_action)
                temp_length.append(int(Length[0].item()))
                
            total_length.append(min(temp_length))
        print(total_length)
        print(f"{H}X{W} Sampling {total_sampling} Times  with Temp = {selecter.T} Mean Locations on {path}",sum(total_length)/len(total_length))  # cost: (batch)
        #print(output_)
    #    print(output[1])  # ll: (batch)

