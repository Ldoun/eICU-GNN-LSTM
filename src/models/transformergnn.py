from tqdm import tqdm
import torch
import torch.nn as nn
from src.models.transformer import define_transformer_encoder
from src.models.pyg_ns import define_ns_gnn_encoder
from src.models.utils import init_weights, get_act_fn


class TransformerGNN(torch.nn.Module):
    """
    model class for Transformer-GNN with node-sampling scheme.
    """
    def __init__(self, config):
        super().__init__()
        #self.input_layer = nn.Linear(config['input_size'], config['feature_size'])
        self.transformer_encoder = define_transformer_encoder()(config)        
        self.gnn_name = config['gnn_name']
        self.gnn_encoder = define_ns_gnn_encoder(config['gnn_name'])(config)
        self.last_act = get_act_fn(config['final_act_fn'])
        self.transformer_out = nn.Linear(config['feature_size'], config['out_dim'])
        self._initialize_weights()

    def _initialize_weights(self):
        init_weights(self.modules())

    def forward(self, x, flat, adjs, batch_size, edge_weight):
        #x = self.input_layer(x) #change dimension of input to match pos encoder
        seq = x.permute(1, 0, 2)
        out, _ = self.transformer_encoder.forward(seq)
        last = out[:, -1, :] if len(out.shape)==3 else out
        last = last[:batch_size]
        out = out.view(out.size(0), -1) # all_nodes, transformer_outdim
        x = out
        x = self.gnn_encoder(x, flat, adjs, edge_weight, last)
        y = self.last_act(x)        
        transformer_y = self.last_act(self.transformer_out(last))
        return y, transformer_y
    
    def infer_transformer_by_batch(self, ts_loader, device):
        transformer_outs = []
        lasts = []
        transformer_ys = []
        for inputs, labels, ids in ts_loader:
            seq, flat = inputs
            seq = seq.to(device)
            #seq = self.input_layer(seq)
            seq = seq.permute(1, 0, 2)
            out, _ = self.transformer_encoder.forward(seq)
            last = out[:, -1, :] if len(out.shape)==3 else out
            out = out.view(out.size(0), -1)
            transformer_y = self.last_act(self.transformer_out(last))
            transformer_outs.append(out)
            lasts.append(last)
            transformer_ys.append(transformer_y)
        transformer_outs = torch.cat(transformer_outs, dim=0) # [entire_g, dim]
        lasts = torch.cat(lasts, dim=0) # [entire_g, dim]
        transformer_ys = torch.cat(transformer_ys, dim=0)
        print('Got all transformer output.')
        return transformer_outs, lasts, transformer_ys
    
    def inference(self, x_all, flat_all, edge_weight, ts_loader, subgraph_loader, device, get_emb=False):
        # first collect transformer outputs by minibatching:
        transformer_outs, last_all, transformer_ys = self.infer_transformer_by_batch(ts_loader, device)

        # then pass transformer outputs to gnn
        x_all = transformer_outs
        out = self.gnn_encoder.inference(x_all, flat_all, subgraph_loader, device, edge_weight, last_all, get_emb=get_emb)

        out = self.last_act(out)

        return out, transformer_ys