import torch
import torch.nn as nn
import math
from encoder import GraphAttentionEncoder
from Env import Env
from sampler import TopKSampler, CategoricalSampler, New_Sampler

from decoder_utils import concat_embedding, concat_graph_embedding
class Decoder_uBRP(nn.Module):
    def __init__(self, 
                 device, 
                 embed_dim=128, 
                 n_encode_layers=3, 
                 n_heads=8, 
                 ff_hidden = 128, 
                 n_containers=8, 
                 max_stacks = 4,
                 max_tiers = 4,
                 return_pi = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.embed_dim = embed_dim
        self.concat_embed_dim = embed_dim*2
        self.total_embed_dim = embed_dim*3
        self.return_pi = return_pi
        self.Encoder = GraphAttentionEncoder(n_heads = n_heads, embed_dim = embed_dim, n_encode_layers = n_encode_layers, max_stacks = max_stacks, max_tiers = max_tiers, ff_hidden=ff_hidden).to(device)
        self.Wq_fixed = nn.Linear(embed_dim, embed_dim*3, bias=False)
        self.Wk_2 = nn.Linear(embed_dim*3, embed_dim*3, bias=False)

        self.W_O = nn.Sequential(
            nn.Linear(embed_dim*3, embed_dim*2),
            nn.ReLU(),
            nn.Linear(embed_dim*2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Linear(embed_dim//2,embed_dim//4),
            nn.ReLU(),
            nn.Linear(embed_dim//4, 1)
            #nn.Linear(256, 512), # It can be More Deeper
            #nn.ReLU(),
            #nn.Linear(512,256),
            #nn.ReLU(),
            #nn.Linear(256,128),
            #nn.ReLU()
        ).to(device)
        

    def compute_dynamic(self, mask, node_embeddings):
        logits = self.W_O(node_embeddings)
        logtis_with_mask = logits - mask.to(torch.int)*1e9
        return logtis_with_mask.squeeze(dim=2)
    def forward(self, x, n_containers=8, return_pi=False, decode_type='sampling', return_ap=False):
        env = Env(self.device,x,self.concat_embed_dim)
        env.find_target_stack()
        encoder_output=self.Encoder(env.x)
        node_embeddings, graph_embedding = encoder_output
        target_embeddings = node_embeddings[torch.arange(node_embeddings.size(0)),env.target_stack,:]
        total_embeddings = concat_graph_embedding(graph_embedding, concat_graph_embedding(target_embeddings, node_embeddings))
        #mask(batch,max_stacks,1) 
        #step_context=target_stack_embedding(batch, 1, embed_dim) 
        mask = env.create_mask_rBRP()
        #default n_samples=1
        batch,max_stacks,max_tiers = x.size()

        logits = self.compute_dynamic(mask, total_embeddings)
        # log_p (batch,max_stacks)

        p = torch.softmax(logits, dim=1).view(batch, max_stacks)
        return logits

