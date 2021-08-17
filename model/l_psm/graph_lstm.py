import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

from utils import import_class, count_params
from model.ms_gcn import MultiScale_GraphConv as MS_GCN
from model.l_psm import *
from utils.LP_logsig import *


class LSTM_GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_gcn_scales, spatial_kernel_size, num_joints, ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        
        self.W_g = MS_GCN(num_gcn_scales, in_channels, 4 * hidden_channels,
                           A_binary, disentangled_agg=True)        
        self.U_g = MS_GCN(num_gcn_scales, hidden_channels, 4 * hidden_channels,
                           A_binary, disentangled_agg=True)
        self.b = nn.Parameter(torch.Tensor(4 * hidden_channels, 1, num_joints))

        self.attention = attention(hidden_channels)

    def forward(self,
                x,
                init_states=None):

        N, C, T, V = x.size()
        hidden_seq = []

        if init_states is None:
            h_t, c_t = (
                torch.zeros(N, self.hidden_channels, 1, V).to(x.device),
                torch.zeros(N, self.hidden_channels, 1, V).to(x.device),
            )
        else:
            h_t, c_t = init_states

        HS = self.hidden_channels
        for t in range(T):
            x_t = x[:, :, t:t + 1, :]

            gates = self.W_g(x_t)[0] + self.U_g(h_t)[0] + self.b
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]),  # input
                torch.sigmoid(gates[:, HS:HS * 2]),  # forget
                torch.tanh(gates[:, HS * 2:HS * 3]),
                torch.sigmoid(gates[:, HS * 3:]),  # output
            )

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            h_t = self.attention(h_t) + h_t

            hidden_seq.append(h_t)

        # reshape hidden_seq p/ retornar
        hidden_seq = torch.cat(hidden_seq, dim=2)

        return hidden_seq, (h_t, c_t)


class attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        c1 = 32
        c2 = 64

        self.W = nn.Linear(in_channels, c1)
        self.W_q = nn.Linear(c1, c2)
        self.W_h = nn.Linear(in_channels, c2)
        self.U_s = nn.Linear(c2, 1)

    def forward(self, h):
        # shape of h is (N, C, 1, V)
        N, C, T, V = h.shape
        x = h.squeeze(2).permute(0, 2, 1).contiguous()
        q = F.relu(self.W(x.sum(axis=1)))
        u = F.tanh(self.W_h(x) + self.W_q(q).unsqueeze(1).expand(-1, V, -1))
        a = F.sigmoid(self.U_s(u)).squeeze(-1)
        a = a.view(-1, 1, 1, V).contiguous()

        return a * h
