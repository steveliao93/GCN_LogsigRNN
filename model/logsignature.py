import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata
import signatory


class LogSig(torch.nn.Module):
    def __init__(self, in_channels, n_segments, logsig_depth, logsig_channels):
        super(LogSig, self).__init__()
        self.in_channels = in_channels
        self.n_segments = n_segments
        self.logsig_depth = logsig_depth

        self.logsignature = signatory.LogSignature(depth=logsig_depth)

        self.logsig_channels = logsig_channels

    def forward(self, inp):
        # inp is a three dimensional tensor of shape (batch, stream, in_channels)
        nT = inp.size(1)
        dim_path = inp.size(-1)
        t_vec = np.linspace(1, nT, self.n_segments + 1)
        t_vec = [int(round(x)) for x in t_vec]

        MultiLevelLogSig = []
        for i in range(self.n_segments):
            MultiLevelLogSig.append(self.logsignature(
                inp[:, t_vec[i] - 1:t_vec[i + 1], :].clone()).unsqueeze(1))
#         print(MultiLevelLogSig.type())
        out = torch.cat(MultiLevelLogSig, axis=1)
        return out




class sp(torch.nn.Module):
    def __init__(self, n_segments):
        super(sp, self).__init__()

        self.n_segments = n_segments

    def forward(self, inp):
        nT = inp.size(1)
        t_vec = np.linspace(1, nT, self.n_segments + 1)
        t_vec = [int(round(x)) - 1 for x in t_vec]
        return inp[:, t_vec[:-1]].clone()


def get_time_vector(size, length):
    return torch.linspace(0, 1, length).reshape(1, -1, 1).repeat(size, 1, 1)


def add_time(x, device):
    t = get_time_vector(x.shape[0], x.shape[1]).to(device)
    return torch.cat([x, t], dim=2)


