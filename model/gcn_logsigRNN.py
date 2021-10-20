import sys
sys.path.insert(0, '')

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import import_class, count_params
from model.gcn import GraphConv as GCN
from model.l_psm import *


class Model(nn.Module):
    def __init__(self,
                 num_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 graph,
                 in_channels=3):
        super(Model, self).__init__()

        Graph = import_class(graph)
        A_binary = Graph().A_binary

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        # channels
        c1 = 90
        c2 = 180     
        self.c1 = c1
        self.c2 = c2

        self.gcn1 = GCN(num_gcn_scales, 3, c1,
                           A_binary)

        self.n_segments1 = 50
        self.logsig_channels1 = signatory.logsignature_channels(in_channels=c1,
                                                                depth=2)
        self.logsig1 = LogSig_v2(c1, logsig_depth=2,
                                 logsig_channels=self.logsig_channels1)
        self.start_position1 = sp_v2()

        self.lstm1 = nn.LSTM(
            input_size=c1 + self.logsig_channels1,
            hidden_size=c1,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        self.gcn2 = GCN(num_gcn_scales, c1, c2,
                           A_binary)

        self.n_segments2 = 30
        self.logsig_channels2 = signatory.logsignature_channels(in_channels=c2,
                                                                depth=2)
        self.logsig2 = LogSig_v2(c2, logsig_depth=2,
                                 logsig_channels=self.logsig_channels2)
        self.start_position2 = sp_v2()

        self.lstm2 = nn.LSTM(
            input_size=c2 + self.logsig_channels2,
            hidden_size=c2,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        self.fc = nn.Linear(c2, num_class)

    def forward(self, x):
        N, C, T, V, M = x.shape

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N * M, V, C, T).permute(0, 2, 3, 1).contiguous()
        # N,C,T,V

        x = F.relu(self.gcn1(x), inplace=False)
        x = x.permute(0, 3, 2, 1).contiguous().view(
            N * M * V, T, self.c1).contiguous()

        x_sp = self.start_position1(x,self.n_segments1).type_as(x)
        x_logsig = self.logsig1(x, self.n_segments1).type_as(x)
        self.lstm1.flatten_parameters()
        x, _ = self.lstm1(torch.cat([x_logsig, x_sp], axis=-1))
        x = nn.BatchNorm1d(self.n_segments1).to(x.device)(x)
        x = x.view(
            N * M, V, self.n_segments1, self.c1).permute(0, 3, 2, 1).contiguous()

        x = F.relu(self.gcn2(x), inplace=False)
        x = x.permute(0, 3, 2, 1).contiguous().view(
            N * M * V, self.n_segments1, self.c2).contiguous()

        x_sp = self.start_position2(x, self.n_segments2).type_as(x)
        x_logsig = self.logsig2(x, self.n_segments2).type_as(x)
        self.lstm2.flatten_parameters()
        x, _ = self.lstm2(torch.cat([x_logsig, x_sp], axis=-1))
        x = nn.BatchNorm1d(self.n_segments2).to(x.device)(x)
        x = x.view(
            N * M, V, self.n_segments2, self.c2).permute(0, 3, 2, 1).contiguous()
        
        out = x
        out_channels = out.size(1)
        out = out.view(N, M, out_channels, -1)
        out = out.mean(3)   
        out = out.mean(1)   

        out = self.fc(out)
        return out

