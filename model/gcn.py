import sys
sys.path.insert(0, '')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from graph.tools import k_adjacency, normalize_adjacency_matrix
from model.activation import activation_factory


class GraphConv(nn.Module):
    def __init__(self,
                 num_scales,
                 in_channels,
                 out_channels,
                 A_binary,
                 activation='relu'):
        super().__init__()
        self.num_scales = num_scales

        A_powers = [k_adjacency(A_binary, k, with_self=True) for k in range(num_scales)]
        A_powers = np.concatenate([normalize_adjacency_matrix(g) for g in A_powers])

        self.A_powers = torch.Tensor(A_powers)
        self.A_res = nn.init.uniform_(nn.Parameter(torch.Tensor(self.A_powers.shape)), -1e-6, 1e-6)

        self.mlp = MLP(in_channels * num_scales, [out_channels], activation=activation)

    def forward(self, x):
        N, C, T, V = x.shape
        self.A_powers = self.A_powers.to(x.device)
        A = self.A_powers.to(x.dtype)
        A = A + self.A_res.to(x.dtype)
        support = torch.einsum('vu,nctu->nctv', A, x)
        support = support.view(N, C, T, self.num_scales, V)
        support = support.permute(0,3,1,2,4).contiguous().view(N, self.num_scales*C, T, V)
        out = self.mlp(support)
        return out

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        channels = [in_channels] + out_channels
        self.layers = nn.ModuleList()
        for i in range(1, len(channels)):
            self.layers.append(nn.Conv2d(channels[i-1], channels[i], kernel_size=1))
            self.layers.append(nn.BatchNorm2d(channels[i]))
            self.layers.append(activation_factory(activation))

    def forward(self, x):
        # Input shape: (N,C,T,V)
        for layer in self.layers:
            x = layer(x)
        return x
