import os
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import math
import warnings



def get_checkpoint_path(model_dir):
    """Return path of latest checkpoint found in the model directory."""
    chkpt =  str(list(Path(model_dir).glob('checkpoints/*'))[-1])
    return chkpt


def define_loss_fn(config):
    """define loss function"""
    if config['classification']:
        if config['class_weights'] is False:
            return nn.CrossEntropyLoss()
        else:
            return nn.CrossEntropyLoss(weight=torch.from_numpy(config['class_weights']).float())
    else:
        return nn.MSELoss()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stack_outputs_with_mask(outputs, multi_gpu, val_nid=None):
    log_dict = {}
    for loss_type in outputs[0]:
        if multi_gpu:
            collect = []
            for output in outputs:
                for v in output[loss_type]:
                    collect.append(v)
        else:
            collect = [v[loss_type] for v in outputs]
        if 'loss' in loss_type:
            log_dict[loss_type] =  torch.Tensor(collect)
        else:
            log_dict[loss_type] =  torch.cat(collect).squeeze()
    # mask out
    if val_nid is not None:
        tmp = log_dict[loss_type]
        val_nid = val_nid.type_as(tmp)
        mask  = [n in val_nid for n in log_dict['ids']]
        anti_mask  = [n not in val_nid for n in log_dict['ids']]
        val_dict = {name: log_dict[name][mask] for name in log_dict}
        train_log_dict = {name: log_dict[name][anti_mask] for name in log_dict}
    else:
        val_dict = log_dict
        train_log_dict = log_dict
    return val_dict, train_log_dict


def collect_outputs(outputs, multi_gpu):
    """collect outputs from pytorch-lightning layout"""
    log_dict = {}
    for loss_type in outputs[0]:
        if loss_type in ['pred', 'truth', 'ids', 'pred_lstm', 'hid', 'edge_index', 'edge_attn_1', 'edge_attn_2']:
            if multi_gpu:
                collect = []
                for output in outputs:
                    for v in output[loss_type]:
                        collect.append(v)
            else:
                collect = [v[loss_type] for v in outputs]
            log_dict[loss_type] =  torch.cat(collect).squeeze().detach().cpu().numpy()
        else:
            if multi_gpu:
                collect = []
                for output in outputs:
                    for v in output[loss_type]:
                        if v == v:
                            collect.append(v)
            else:
                collect = [v[loss_type] for v in outputs if v[loss_type] == v[loss_type]]
            if collect:
                log_dict[loss_type] = torch.stack(collect).mean().detach().cpu().numpy()
            else:
                log_dict[loss_type] = float('nan')
    return log_dict


def init_weights(modules):
    """initialize model weights"""
    for m in modules:
        if isinstance(m, nn.LSTM):
            nn.init.xavier_uniform_(m.weight_ih_l0)
            nn.init.orthogonal_(m.weight_hh_l0)
            for names in m._all_weights:
                for name in filter(lambda n: 'bias' in n, names):
                    bias = getattr(m, name)
                    n = bias.size(0)
                    start, end = n // 4, n // 2
                    bias.data[start:end].fill_(1.0)
        elif isinstance(m, nn.Linear):
            n = m.weight.size(1)
            m.weight.data.normal_(0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()


def get_act_fn(name):
    """define activation function"""
    if name is None:
        act_fn = lambda x: x
    elif name == 'relu':
        act_fn = nn.ReLU()
    elif name == 'leakyrelu':
        act_fn = nn.LeakyReLU()
    elif name == 'hardtanh':
        act_fn = nn.Hardtanh(min_val=1 / 48, max_val=100)
    return act_fn



def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)