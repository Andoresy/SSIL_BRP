from data import generate_data
import torch
from Env import Env
import gc
from sampler import TopKSampler,CategoricalSampler, New_Sampler
from tqdm import tqdm
import time
import numpy as np
import random
import copy
from scipy.stats import ttest_rel
class SSIL_baseline:
    def __init__(self, device, init_model,  max_stacks = 6, max_tiers = 8, n_problems = 1000, alpha_c = 0.05, alpha_r = 0.1):
        self.baseline_model = copy.deepcopy(init_model)
        self.baseline_model.eval()
        self.device = device
        self.alpha_c = alpha_c
        self.alpha_r = alpha_r
        self.problems = generate_data(device, n_problems, n_containers = max_stacks*(max_tiers-2), max_stacks = max_stacks, max_tiers = max_tiers )
    def callback(self, device = 'cuda:0', model = None,  max_stacks = 6, max_tiers = 8, epoch=None): #Callback - Commit in 
        """ Callback - Commit (In paper)
        """
        if model is None:
            raise NotImplementedError
        model.eval()
        problems = self.problems
        baseline_score = self.rollout(self.baseline_model, max_stacks, max_tiers, problems).cpu()
        model_score = self.rollout(model, max_stacks, max_tiers, problems).cpu()
        t, p = ttest_rel(baseline_score, model_score)  # scipy.stats.ttest_rel
        b_m, m_m = torch.mean(baseline_score).item(), torch.mean(model_score).item()
        print(f"Epoch{epoch} Evaluate: ")
        print(f"Score: {torch.mean(model_score).item():.4f}   Baseline Score: {torch.mean(baseline_score).item():.4f}" )
        p_val = p / 2
        alpha = self.alpha_c
        if b_m > m_m:
            print(f'p-value: {p_val}')
            if p_val < alpha:
                print('Update baseline')
                self.baseline_model = copy.deepcopy(model)
        return p_val < alpha and b_m>m_m
    def callback_rollback(self, device = 'cuda:0', model = None,  max_stacks = 6, max_tiers = 8, epoch=None):
        """ Callback - Rollback (In paper)
        """
        model.eval()
        problems = self.problems
        if model is None:
            raise NotImplementedError
        baseline_score = self.rollout(self.baseline_model, max_stacks, max_tiers, problems).cpu()
        model_score = self.rollout(model, max_stacks, max_tiers, problems).cpu()
        t, p = ttest_rel(baseline_score, model_score)  # scipy.stats.ttest_rel
        b_m, m_m = torch.mean(baseline_score).item(), torch.mean(model_score).item()
        p_val = p / 2
        alpha = self.alpha_r
        return p_val < alpha and b_m < m_m
    def rollout(self, model, max_stacks, max_tiers, problems):
        selecter = TopKSampler()
        env = Env(device = self.device, x = problems)
        env.clear()
        Length = torch.zeros(len(problems)).to( self.device)
        for _ in range(200):
            if env.all_empty():
                break
            output = model(env.x)
            next_action = selecter(output)
            source_node, dest_node = next_action//max_stacks, next_action%max_stacks
            actions = torch.cat((source_node,dest_node), 1)
            Length += (1.0 - env.empty.type(torch.float64))
            env.step(actions)
        return Length
    def create_ss_samples(self, device = 'cuda:0', max_stacks = 6, max_tiers = 8, n_problems = 1000,n_samplings = 128):
        """ Create Self-Supervised Learning Samples with ESS
        """
        model = self.baseline_model
        model.eval()
        if model is None:
            raise NotImplementedError
        problems = generate_data(device, n_problems, n_containers = max_stacks*(max_tiers-2), max_stacks = max_stacks, max_tiers = max_tiers )
        train_data = []
        label_data = []
        selecter = New_Sampler(T=1)
        sampling_batch = n_samplings
        for i in tqdm(range(len(problems)), desc='Sampling Self-Supervised Learning Trajectories'):
            gc.collect()
            env = Env(device = device, x = problems[i:i+1].repeat(sampling_batch,1,1))
            env.clear()
            trajectory = []
            traj_actions = []
            for step in range(500):
                trajectory.append(env.x.clone())
                if env.empty.any().item(): # If any state is at terminal 
                    index = env.empty.long().nonzero(as_tuple=False)[0]
                    best_traj = torch.cat(trajectory)[torch.arange(index.item(), index.item()+sampling_batch*(step), sampling_batch),:,:].clone()
                    best_action = torch.cat(traj_actions)[torch.arange(index.item(), index.item()+sampling_batch*(step), sampling_batch)].clone()
                    train_data.append(best_traj)
                    label_data.append(best_action)
                    break
                output = model(env.x)
                next_action = selecter(output)
                traj_actions.append(next_action.view(-1).clone())
                source_node, dest_node = next_action//max_stacks, next_action%max_stacks
                actions = torch.cat((source_node,dest_node), 1)
                env.step(actions)
            del trajectory[:]
        train_data = torch.cat(train_data).to(device)
        label_data = torch.cat(label_data).to(device)
        return train_data, label_data