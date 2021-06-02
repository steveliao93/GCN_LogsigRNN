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


class MS_G3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_size,
                 window_stride,
                 window_dilation,
                 embed_factor=1,
                 activation='relu'):
        super().__init__()
        self.window_size = window_size
        self.out_channels = out_channels
        self.embed_channels_in = self.embed_channels_out = out_channels // embed_factor
        if embed_factor == 1:
            self.in1x1 = nn.Identity()
            self.embed_channels_in = self.embed_channels_out = in_channels
            # The first STGC block changes channels right away; others change at collapse
            if in_channels == 3:
                self.embed_channels_out = out_channels
        else:
            self.in1x1 = MLP(in_channels, [self.embed_channels_in])

        self.gcn3d = nn.Sequential(
            UnfoldTemporalWindows(window_size, window_stride, window_dilation),
            SpatialTemporal_MS_GCN(
                in_channels=self.embed_channels_in,
                out_channels=self.embed_channels_out,
                A_binary=A_binary,
                num_scales=num_scales,
                window_size=window_size,
                use_Ares=True
            )
        )

        self.out_conv = nn.Conv3d(
            self.embed_channels_out, out_channels, kernel_size=(1, self.window_size, 1))
        self.out_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        N, _, T, V = x.shape
        x = self.in1x1(x)
        # Construct temporal windows and apply MS-GCN
        x = self.gcn3d(x)

        # Collapse the window dimension
        x = x.view(N, self.embed_channels_out, -1, self.window_size, V)
        x = self.out_conv(x).squeeze(dim=3)
        x = self.out_bn(x)

        # no activation
        return x


class MultiWindow_MS_G3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_sizes=[3, 5],
                 window_stride=1,
                 window_dilations=[1, 1]):

        super().__init__()
        self.gcn3d = nn.ModuleList([
            MS_G3D(
                in_channels,
                out_channels,
                A_binary,
                num_scales,
                window_size,
                window_stride,
                window_dilation
            )
            for window_size, window_dilation in zip(window_sizes, window_dilations)
        ])

    def forward(self, x):
        # Input shape: (N, C, T, V)
        out_sum = 0
        for gcn3d in self.gcn3d:
            out_sum += gcn3d(x)
        # no activation
        return out_sum


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
        c1 = 96
        c2 = c1 * 2     # 192
        c3 = 288     #
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3

        # r=3 STGC blocks
        self.gcn3d1 = MultiWindow_MS_G3D(
            3, c1, A_binary, num_g3d_scales, window_stride=1)
        self.sgcn1 = nn.Sequential(
            MS_GCN(num_gcn_scales, 3, c1, A_binary, disentangled_agg=True),
            MS_TCN(c1, c1),
            MS_TCN(c1, c1))
        self.sgcn1[-1].act = nn.Identity()
        self.tcn1 = MS_TCN(c1, c1)

        self.gcn1 = MS_GCN(num_gcn_scales, 3, c1,
                           A_binary, disentangled_agg=True)

        self.n_segments1 = 80
        self.logsig_channels1 = signatory.logsignature_channels(in_channels=c1,
                                                                depth=2)
        self.logsig1 = LogSig_v1(c1, n_segments=self.n_segments1, logsig_depth=2,
                                 logsig_channels=self.logsig_channels1)
        self.start_position1 = sp(self.n_segments1)

        self.lstm1 = nn.LSTM(
            input_size=c1 + self.logsig_channels1,
            hidden_size=c1,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.logsig_bn1 = nn.BatchNorm1d(self.n_segments1)

        self.gcn2 = MS_GCN(num_gcn_scales, c1, c2,
                           A_binary, disentangled_agg=True)

        self.n_segments2 = 80
        self.logsig_channels2 = signatory.logsignature_channels(in_channels=c2,
                                                                depth=2)
        self.logsig2 = LogSig_v1(c2, n_segments=self.n_segments2, logsig_depth=2,
                                 logsig_channels=self.logsig_channels2)
        self.start_position2 = sp(self.n_segments2)

        self.lstm2 = nn.LSTM(
            input_size=c2,
            hidden_size=c2,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.logsig_bn2 = nn.BatchNorm1d(self.n_segments2)

        self.gcn3 = MS_GCN(num_gcn_scales, c2, c3,
                           A_binary, disentangled_agg=True)

        self.n_segments3 = 20
        self.logsig_channels3 = signatory.logsignature_channels(in_channels=c3,
                                                                depth=2)
        self.logsig3 = LogSig_v1(c3, n_segments=self.n_segments3, logsig_depth=2,
                                 logsig_channels=self.logsig_channels3)
        self.start_position3 = sp(self.n_segments3)

        self.lstm3 = nn.LSTM(
            input_size=self.logsig_channels3 + c3,
            hidden_size=c3,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.logsig_bn3 = nn.BatchNorm1d(self.n_segments3)

        self.gcn3d2 = MultiWindow_MS_G3D(
            c1, c2, A_binary, num_g3d_scales, window_stride=2)
        self.sgcn2 = nn.Sequential(
            MS_GCN(num_gcn_scales, c1, c1, A_binary, disentangled_agg=True),
            MS_TCN(c1, c2, stride=2),
            MS_TCN(c2, c2))
        self.sgcn2[-1].act = nn.Identity()
        self.tcn2 = MS_TCN(c2, c2)

        self.gcn3d3 = MultiWindow_MS_G3D(
            c2, c3, A_binary, num_g3d_scales, window_stride=2)
        self.sgcn3 = nn.Sequential(
            MS_GCN(num_gcn_scales, c2, c2, A_binary, disentangled_agg=True),
            MS_TCN(c2, c3, stride=2),
            MS_TCN(c3, c3))
        self.sgcn3[-1].act = nn.Identity()
        self.tcn3 = MS_TCN(c3, c3)

        self.fc = nn.Linear(c2, num_class)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N * M, V, C, T).permute(0, 2, 3, 1).contiguous()
        # N,C,T,V

        x = F.relu(self.gcn1(x), inplace=False)
        x = x.permute(0, 3, 2, 1).contiguous().view(
            N * M * V, T, self.c1).contiguous()

        x_sp = self.start_position1(x).type_as(x)
        x_logsig = self.logsig1(x).type_as(x)
        self.lstm1.flatten_parameters()
        x, _ = self.lstm1(torch.cat([x_logsig, x_sp], axis=-1))
        x = self.logsig_bn1(x)
        x = x.view(
            N * M, V, self.n_segments1, self.c1).permute(0, 3, 2, 1).contiguous()

        x = F.relu(self.gcn2(x), inplace=False)
        x = x.permute(0, 3, 2, 1).contiguous().view(
            N * M * V, self.n_segments1, self.c2).contiguous()

        #x_sp = self.start_position2(x).type_as(x)
        self.lstm2.flatten_parameters()
        x, _ = self.lstm2(x)
        x = self.logsig_bn2(x)
        x = x.view(
            N * M, V, self.n_segments2, self.c2).permute(0, 3, 2, 1).contiguous()

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
