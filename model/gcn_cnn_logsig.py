import sys
sys.path.insert(0, '')

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import import_class, count_params
from model.ms_gcn import MultiScale_GraphConv as MS_GCN
from model.ms_tcn import MultiScale_TemporalConv as MS_TCN
from model.ms_gtcn import SpatialTemporal_MS_GCN, UnfoldTemporalWindows
from model.mlp import MLP
from model.activation import activation_factory
from model.l_psm import *


class Model(nn.Module):
    def __init__(self,
                 num_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 graph,
                 in_channels=3):
        super(Model, self).__init__()

        Graph = import_class(graph)
        A_binary = Graph().A_binary

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        # channels
        c1 = 288
        self.n_stream = 2
        c2 = 48

        self.c1 = c1
        self.c2 = c2

        # r=3 STGC blocks

        self.gcn1 = MS_GCN(num_gcn_scales, 3, c1,
                           A_binary, disentangled_agg=True)

        self.conv1 = nn.Conv1d(c1, self.n_stream * c2, 1)

        self.n_segments1 = 80
        self.logsig_channels1 = signatory.logsignature_channels(in_channels=c2,
                                                                depth=2)
        self.logsig1 = LogSig_v1(c1, n_segments=self.n_segments1, logsig_depth=2,
                                 logsig_channels=self.logsig_channels1)
        self.start_position1 = sp(self.n_segments1)

        self.lstm1 = nn.LSTM(
            input_size=self.logsig_channels1 + c2,
            hidden_size=c2,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.logsig_bn1 = nn.BatchNorm1d(self.n_segments1)

        self.fc = nn.Linear(c2 * self.n_stream, num_class)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N * M, V, C, T).permute(0, 2, 3, 1).contiguous()
        # N,C,T,V

        x = F.relu(self.gcn1(x), inplace=False)
        x = x.permute(0, 3, 2, 1).contiguous().view(
            N * M * V, T, self.c1).contiguous()
        x = x.permute(0, 2, 1).contiguous()
        x = self.conv1(x).permute(0, 2, 1).contiguous()

        x_stream = []
        for i in range(self.n_stream):
            y = x[:, :, i * self.c2:(i + 1) * self.c2].clone()
            y_sp = self.start_position1(y).type_as(x)

            y_logsig = self.logsig1(y).type_as(x)
            self.lstm1.flatten_parameters()

            y, _ = self.lstm1(torch.cat([y_logsig, y_sp], axis=-1))
            #x = self.dropout1(x)
            y = self.logsig_bn1(y)
            y = y.view(
                N * M, V, self.n_segments1, self.c1).permute(0, 3, 2, 1).contiguous()
            x_stream.append(y)
        x = torch.cat(x_stream, axis=1)

        # Apply activation to the sum of the pathways
        # x = F.relu(self.sgcn1(x) + self.gcn3d1(x), inplace=True)
        # x = self.tcn1(x)

        # x = F.relu(self.sgcn2(x) + self.gcn3d2(x), inplace=True)
        # x = self.tcn2(x)

        # x = F.relu(self.sgcn3(x) + self.gcn3d3(x), inplace=True)
        # x = self.tcn3(x)

        out = x
        out_channels = out.size(1)
        out = out.view(N, M, out_channels, -1)
        out = out.mean(3)   # Global Average Pooling (Spatial+Temporal)
        out = out.mean(1)   # Average pool number of bodies in the sequence

        out = self.fc(out)
        return out


if __name__ == "__main__":
    # For debugging purposes
    import sys
    sys.path.append('..')

    model = Model(
        num_class=60,
        num_point=25,
        num_person=2,
        num_gcn_scales=13,
        num_g3d_scales=6,
        graph='graph.ntu_rgb_d.AdjMatrixGraph'
    )

    N, C, T, V, M = 6, 3, 50, 25, 2
    x = torch.randn(N, C, T, V, M)
    model.forward(x)

    print('Model total # params:', count_params(model))
