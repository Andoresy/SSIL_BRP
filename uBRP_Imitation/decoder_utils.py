import torch
import torch.nn.functional as F

def concat_embedding(node_embeddings, device='cuda:0'):
    
    batch_size, width, embed_size = node_embeddings.size()

    # Pairwise Concatenation
    reshaped_node_embeddings = node_embeddings.view(batch_size, 1, width, embed_size)
    reshaped_node_embeddings = reshaped_node_embeddings.expand(batch_size, width, width, embed_size)
    newnode_embeddings = torch.cat([reshaped_node_embeddings, reshaped_node_embeddings.transpose(1, 2)], dim=3)

    # Size reformulation
    newnode_embeddings = newnode_embeddings.view(batch_size, width * width, embed_size * 2)
    newnode_embeddings = newnode_embeddings.view(batch_size, width, width, embed_size * 2)
    newnode_embeddings = newnode_embeddings.transpose(1, 2).contiguous().view(batch_size, width * width, embed_size * 2)
    return newnode_embeddings.to(device)
def concat_graph_embedding(graph_embedding, node_embedding, device = 'cuda:0'):
    #Concat the graph embedding (i.e. mean stack embedding)
    batch_size, width, embed_size = node_embedding.size()
    _, embed_size_graph = graph_embedding.size()
    extd_grpah_embedding = graph_embedding.view(batch_size, 1, embed_size_graph).repeat([1, width, 1])
    return torch.cat([extd_grpah_embedding, node_embedding], dim=2).to(device)
