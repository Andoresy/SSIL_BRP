import torch
import torch.nn as nn
from encoder import GraphAttentionEncoder
from Env import Env

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
                 **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.embed_dim = embed_dim
        self.concat_embed_dim = embed_dim*2
        self.total_embed_dim = embed_dim*3
        self.Encoder = GraphAttentionEncoder(n_heads = n_heads, embed_dim = embed_dim, n_encode_layers = n_encode_layers, max_stacks = max_stacks, max_tiers = max_tiers, ff_hidden=ff_hidden).to(device)
        self.Wq_fixed = nn.Linear(embed_dim, embed_dim*3, bias=False)
        self.Wk_2 = nn.Linear(embed_dim*3, embed_dim*3, bias=False)

        self.W_O = nn.Sequential( # MLP in paper
            nn.Linear(embed_dim*3, embed_dim*2),
            nn.ReLU(),
            nn.Linear(embed_dim*2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Linear(embed_dim//2,embed_dim//4),
            nn.ReLU(),
            nn.Linear(embed_dim//4, 1)
        ).to(device)
        
    def compute_logits(self, mask, node_embeddings):
        logits = self.W_O(node_embeddings)
        logtis_with_mask = logits - mask.to(torch.int)*1e9
        return logtis_with_mask.squeeze(dim=2)
    def forward(self, x):
        env = Env(self.device,x,self.concat_embed_dim)
        #Encoder
        encoder_output=self.Encoder(env.x)
        node_embeddings, graph_embedding = encoder_output
        #Concat&Mask
        concat_node_embeddings = concat_embedding(node_embeddings, device = self.device)
        total_embeddings = concat_graph_embedding(graph_embedding, concat_node_embeddings, device= self.device)
        mask = env.create_mask_uBRP()
        
        #Decoder
        logits = self.compute_logits(mask, total_embeddings)
        #p = torch.softmax(logits, dim=1).view(batch, max_stacks* max_stacks)
        return logits

