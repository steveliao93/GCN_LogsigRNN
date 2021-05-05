import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph
from net.utils.LP_logsig import *
from net.utils.augmentations import *


class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32,
                         requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        self.c1 = 96
        self.gcn1 = ConvTemporalGraphical(in_channels, self.c1,
                                          spatial_kernel_size)
        # self.FC_1 = nn.Linear(in_channels, self.c1)
        self.n_segments1 = 80
        self.logsig_channels1 = signatory.logsignature_channels(in_channels=self.c1,
                                                                depth=2)
        self.logsig1 = LogSig_v1(self.c1, n_segments=self.n_segments1, logsig_depth=2,
                                 logsig_channels=self.logsig_channels1)
        self.start_position1 = sp(self.n_segments1)

        self.dropout = nn.Dropout(0.5)

        self.lstm1 = nn.LSTM(
            input_size=self.logsig_channels1 + self.c1,
            hidden_size=self.c1,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.logsig_bn1 = nn.BatchNorm1d(self.n_segments1)

        self.lstm_gcn = LSTM_GCN(
            self.c1, 256, spatial_kernel_size, num_joints=25)

        # fcn for prediction
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N * M, V, C, T)
        x = x.permute(0, 2, 3, 1).contiguous()
        # x = x.view(N * M, C, T, V)
        # x = x.view(N * M * V, T, C)
        # x = self.FC_1(x)
        x = self.gcn1(x, self.A)[0]
        x = x.permute(0, 3, 2, 1).contiguous()
        x = x.view(N * M * V, T, self.c1)

        # forwad
        x_sp = self.start_position1(x).type_as(x)
        x_logsig = self.logsig1(x).type_as(x)
        x_logsig = self.dropout(x_logsig)
        self.lstm1.flatten_parameters()
        x, _ = self.lstm1(torch.cat([x_logsig, x_sp], axis=-1))
        x = self.logsig_bn1(x)
        x = x.view(
            N * M, V, self.n_segments1, self.c1).permute(0, 3, 2, 1).contiguous()
        x, _ = self.lstm_gcn(x, self.A)
        # x = self.dropout(x)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x


class LSTM_GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, spatial_kernel_size, num_joints):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        self.W_g = ConvTemporalGraphical(in_channels, 4 * hidden_channels,
                                         spatial_kernel_size)
        self.U_g = ConvTemporalGraphical(hidden_channels, 4 * hidden_channels,
                                         spatial_kernel_size)
        self.b = nn.Parameter(torch.Tensor(4 * hidden_channels, 1, num_joints))

        self.attention = attention(hidden_channels)

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_channels)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self,
                x, A,
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

            gates = self.W_g(x_t, A)[0] + self.U_g(h_t, A)[0] + self.b
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
