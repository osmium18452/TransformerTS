import torch
from torch import nn
from Models.embed import DataEmbedding


class Transformer(nn.Module):
    def __init__(self, input_size,output_size, d_model=512, n_heads=8, num_enc_layers=6, num_dec_layers=6, batch_first=True):
        super(Transformer, self).__init__()
        self.n_heads = n_heads
        self.transformer = nn.Transformer(d_model=d_model, batch_first=batch_first, nhead=n_heads,
                                          num_encoder_layers=num_enc_layers, num_decoder_layers=num_dec_layers)
        self.linear = nn.Linear(d_model, output_size)
        self.enc_embedding = DataEmbedding(input_size, d_model)
        self.dec_embedding = DataEmbedding(input_size, d_model)

    def forward(self, src, tgt):
        # src=torch.tile(src,(1,1,self.n_heads))
        # tgt=torch.tile(tgt,(1,1,self.n_heads))
        # print(src.shape,tgt.shape)
        # exit()
        enc_input = self.enc_embedding(src)
        dec_input = self.dec_embedding(tgt)
        output = self.transformer(enc_input,dec_input)
        output = self.linear(output)
        return output

class Transformer_v1(nn.Module):
    def __init__(self, input_size, output_size, n_heads=8, num_enc_layers=6, num_dec_layers=6, batch_first=True):
        super(Transformer_v1, self).__init__()
        # print("d_model",input_size*n_heads)
        # print(input_size,n_heads)
        self.n_heads = n_heads
        self.transformer = nn.Transformer(d_model=input_size * n_heads, batch_first=batch_first, nhead=n_heads,
                                          num_encoder_layers=num_enc_layers, num_decoder_layers=num_dec_layers)
        self.linear = nn.Linear(input_size * n_heads, output_size)

    def forward(self, src, tgt):
        src = torch.tile(src, (1, 1, self.n_heads))
        tgt = torch.tile(tgt, (1, 1, self.n_heads))
        # print(src.shape,tgt.shape)
        # exit()
        # print("src, tgt shape",src.shape, tgt.shape)
        output = self.transformer(src, tgt)
        output = self.linear(output)
        return output