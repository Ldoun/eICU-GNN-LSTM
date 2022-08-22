import torch
import torch.nn as nn
from src.models.utils import init_weights, get_act_fn, trunc_normal_
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
        self.transformer_pooling = config['transformer_pooling'] 
        self.pos_encoder = PositionalEncoding(config['feature_size'])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config['feature_size'])) # not using trunc_normal
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=config['feature_size'], nhead=config['n_head'], dropout=config['transformer_dropout']) #batch_first=True not possible
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=config['num_layers'])
        
        if config['fp_attn_transformer']:
            self.attn_gradients = {}
            self.attention_map = {}

            for n, encoder_layer in enumerate(self.transformer_encoder.layers):
                encoder_layer.self_attn.register_forward_hook(self.save_attention_map(n))

    def save_attn_gradients(self, n):
        def fn(attn_gradients):
            self.attn_gradients[n] = attn_gradients
        return fn

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, n):
        def fn(_, __, output):
            output[1].register_hook(self.save_attn_gradients(n))
            self.attention_map[n] = output[1]
        return fn

    def get_attention_map(self):
        return self.attention_map
                
    def forward(self, src):
        cls_tokens = self.cls_token.expand(-1, src.shape[1], -1)
        src = torch.cat((cls_tokens, src), dim=0)
        mask = self._generate_square_subsequent_mask(len(src)).cuda()
        self.src_mask = mask
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src,self.src_mask)#, self.src_mask)
        output = torch.transpose(output, 0, 1).contiguous()
        
        if self.transformer_pooling == 'mean':
            output = torch.mean(output, 1).squeeze()
        elif self.transformer_pooling == 'max':
            output = torch.max(output, 1)[0].squeeze()
        elif self.transformer_pooling == 'last':
            output = output[:, -1, :]
        elif self.transformer_pooling == 'first':
            output = output[:, 0, :]
        elif self.transformer_pooling == 'all':
            pass
        else:
            raise NotImplementedError('only transformer_pooling mean / all for now.')
        return output
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask