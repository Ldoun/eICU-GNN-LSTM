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
        self.transformer_pooling = config['transformer_pooling'] # need to apply pooling, trying out with 'ALL'
        self.pos_encoder = PositionalEncoding(config['feature_size'])
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=config['feature_size'], nhead=config['n_head'], dropout=config['dropout_p'], batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=config['num_layers'])
