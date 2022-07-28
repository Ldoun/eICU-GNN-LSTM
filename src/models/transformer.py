from turtle import forward
import torch
import torch.nn as nn
from src.models.utils import init_weights, get_act_fn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
    
def define_transformer_encoder():
    return TimeSeriesTransformer

class TimeSeriesTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config['device']
        self.transformer_pooling = config['transformer_pooling'] 
        self.pos_encoder = PositionalEncoding(config['feature_size'])
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=config['feature_size'], nhead=config['n_head'], dropout=config['transformer_dropout'], batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=config['num_layers'])
        self.init_weights()
        
    def forward(self, src):
        mask = self._generate_square_subsequent_mask(len(src)).to(self.device)
        self.src_mask = mask
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src,self.src_mask)#, self.src_mask)
        if self.pooling == 'mean':
            output = torch.mean(output, 1).squeeze()
        elif self.pooling == 'max':
            output = torch.max(output, 1)[0].squeeze()
        elif self.pooling == 'last':
            output = output[:, -1, :]
        elif self.pooling == 'all':
            pass
        else:
            raise NotImplementedError('only pooling mean / all for now.')
        return output
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask