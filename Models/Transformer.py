import torch
from torch import nn


class Transformer(nn.Module):
    def __init__(self, input_size, n_heads=8, num_enc_layers=6, num_dec_layers=6, batch_first=True):
        super(Transformer, self).__init__()
        self.n_heads=n_heads
        self.transformer = nn.Transformer(d_model=input_size*n_heads, batch_first=True, nhead=n_heads,
                                          num_encoder_layers=num_enc_layers,num_decoder_layers=num_dec_layers)
        self.linear=nn.Linear(input_size*n_heads,input_size)

    def forward(self,src,tgt):
        src=torch.tile(src,(1,1,self.n_heads))
        tgt=torch.tile(tgt,(1,1,self.n_heads))
        # print(src.shape,tgt.shape)
        # exit()
        output=self.transformer(src,tgt)
        output=self.linear(output)
        return output

