import torch
import torch.nn.functional as F

def concat_embedding(node_embeddings, device='cuda:0'):
    # 입력 텐서의 크기를 가져오기
    batch_size, width, embed_size = node_embeddings.size()

    # Width를 서로 concat하여 (Width)^2 크기로 만들기
    reshaped_node_embeddings = node_embeddings.view(batch_size, 1, width, embed_size)
    reshaped_node_embeddings = reshaped_node_embeddings.expand(batch_size, width, width, embed_size)
    newnode_embeddings = torch.cat([reshaped_node_embeddings, reshaped_node_embeddings.transpose(1, 2)], dim=3)

    # 결과를 (batch_size, (width)^2, embed_size*2)로 변환
    newnode_embeddings = newnode_embeddings.view(batch_size, width * width, embed_size * 2)
    newnode_embeddings = newnode_embeddings.view(batch_size, width, width, embed_size * 2)
    newnode_embeddings = newnode_embeddings.transpose(1, 2).contiguous().view(batch_size, width * width, embed_size * 2)
    return newnode_embeddings.to(device)
def concat_graph_embedding(graph_embedding, node_embedding, device = 'cuda:0'):
    batch_size, width, embed_size = node_embedding.size()
    _, embed_size_graph = graph_embedding.size()
    extd_grpah_embedding = graph_embedding.view(batch_size, 1, embed_size_graph).repeat([1, width, 1])
    return torch.cat([extd_grpah_embedding, node_embedding], dim=2).to(device)
if __name__ == "__main__":
    # 예제 입력 생성
    batch_size = 1
    width = 4
    embed_size = 2
    node_embeddings = torch.rand((batch_size, width, embed_size))

    newnode_embeddings = concat_embedding(node_embeddings)

    print("Original node embeddings:")
    print(node_embeddings)
    print("\nConcatenated node embeddings:")
    print(newnode_embeddings)
    print(newnode_embeddings.size())