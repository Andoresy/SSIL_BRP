import torch
import torch.nn as nn


from decoder import Decoder_uBRP

class AttentionModel_LSTM(nn.Module):

    def __init__(self,device, embed_dim=128, n_encode_layers=3, n_heads=8,
                  ff_hidden=128, max_stacks=4,max_tiers=4):
        super().__init__()


        self.Decoder = Decoder_uBRP(device=device,n_encode_layers=n_encode_layers,embed_dim=embed_dim, n_heads=n_heads,ff_hidden=ff_hidden,
                                    max_stacks=max_stacks,max_tiers=max_tiers)
        self.max_stacks=max_stacks
        self.max_tiers=max_tiers
        self.device=device

    def forward(self, x):
        decoder_output = self.Decoder(x)
        return decoder_output


if __name__ == '__main__':

    model = AttentionModel_LSTM('cuda:0')
    
    cnt = 0
    for i, k in model.state_dict().items():
        print(i, k.size(), torch.numel(k))
        cnt += torch.numel(k)
    print('total parameters:', cnt)