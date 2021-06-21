import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata
import signatory


class LogSig_v1(torch.nn.Module):
    def __init__(self, in_channels, n_segments, logsig_depth, logsig_channels):
        super(LogSig_v1, self).__init__()
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


class LogSig_v2(torch.nn.Module):
    def __init__(self, in_channels, logsig_depth, logsig_channels):
        super(LogSig_v2, self).__init__()
        self.in_channels = in_channels
        self.logsig_depth = logsig_depth

        self.logsignature = signatory.LogSignature(depth=logsig_depth)

        self.logsig_channels = logsig_channels

    def forward(self, inp, n_segments):
        # inp is a three dimensional tensor of shape (batch, stream, in_channels)
        nT = inp.size(1)
        dim_path = inp.size(-1)
        t_vec = np.linspace(1, nT, n_segments + 1)
        t_vec = [int(round(x)) for x in t_vec]

        MultiLevelLogSig = []
        for i in range(n_segments):
            MultiLevelLogSig.append(self.logsignature(
                inp[:, t_vec[i] - 1:t_vec[i + 1], :].clone()).unsqueeze(1))
#         print(MultiLevelLogSig.type())
        out = torch.cat(MultiLevelLogSig, axis=1)
        return out


class LogSig_rolling(nn.Module):
    # rolling windows of logsig
    def __init__(self, in_channels, kernel_size, stride, logsig_depth):
        super().__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.logsig_depth = logsig_depth

        self.logsignature = signatory.LogSignature(depth=logsig_depth)

    def forward(self, inp):
        nT = inp.size(1)
        dim_path = inp.size(-1)

        MultiLevelLogSig = []
        for i in range((nT - self.kernel_size) // self.stride + 1):
            MultiLevelLogSig.append(self.logsignature(
                inp[:, i * self.stride:i * self.stride + self.kernel_size, :].clone()).unsqueeze(1))
        out = torch.cat(MultiLevelLogSig, axis=1)
        return out


class LogSig_var(torch.nn.Module):
        # Variable length version of logsig layer
    def __init__(self, in_channels, n_segments, logsig_depth, logsig_channels):
        super(LogSig_var, self).__init__()
        self.in_channels = in_channels
        self.n_segments = n_segments
        self.logsig_depth = logsig_depth

        self.logsignature = signatory.LogSignature(depth=logsig_depth)

        self.logsig_channels = logsig_channels

    def forward(self, inp, length, n_batchs):
        nT = inp.size(1)
        dim_path = inp.size(-1)
        batch_size = int(inp.size(0) / n_batchs)

        MultiLevelLogSig = torch.zeros(
            inp.size(0), self.n_segments, self.logsig_channels)

        for i in range(n_batchs):

            tmp_fm = length[i].item()
            Tmp = inp[i * batch_size:(i + 1) * batch_size, :tmp_fm].clone()
            # Repeat the actions
            if tmp_fm < self.n_segments * 2:
                tmpenlarge = (self.n_segments * 2) // tmp_fm + 1
                tarlen = tmpenlarge * tmp_fm
                Tmp = Tmp.repeat(
                    1, tmpenlarge, 1).reshape(batch_size, tarlen, dim_path).contiguous()
                tmp_fm = tarlen
            t_vec = np.linspace(1, tmp_fm, self.n_segments + 1)
            t_vec = [int(round(x)) for x in t_vec]

            for j in range(self.n_segments):
                MultiLevelLogSig[i * batch_size:(i + 1) * batch_size, j] = self.logsignature(
                    Tmp[:, t_vec[j] - 1:t_vec[j + 1]].unsqueeze(0))
        return MultiLevelLogSig


class sp(torch.nn.Module):
    def __init__(self, n_segments):
        super(sp, self).__init__()

        self.n_segments = n_segments

    def forward(self, inp):
        nT = inp.size(1)
        t_vec = np.linspace(1, nT, self.n_segments + 1)
        t_vec = [int(round(x)) - 1 for x in t_vec]
        return inp[:, t_vec[:-1]].clone()


class sp_v2(torch.nn.Module):
    def __init__(self, ):
        super(sp_v2, self).__init__()

    def forward(self, inp, n_segments):
        nT = inp.size(1)
        t_vec = np.linspace(1, nT, n_segments + 1)
        t_vec = [int(round(x)) - 1 for x in t_vec]
        return inp[:, t_vec[:-1]].clone()


class sp_var(torch.nn.Module):
    def __init__(self, n_segments):
        super(sp_var, self).__init__()

        self.n_segments = n_segments

    def forward(self, inp, length, n_batchs):
        batch_size = int(inp.size(0) / n_batchs)
        out = []
        for i in range(n_batchs):
            tmp_fm = length[i].item()
            if tmp_fm < self.n_segments * 2:
                tmpenlarge = (self.n_segments * 2) // tmp_fm + 1
                tmp_fm = tmpenlarge * tmp_fm
            t_vec = np.linspace(1, tmp_fm, self.n_segments + 1)
            t_vec = [int(round(x)) - 1 for x in t_vec]
            out.append(inp[i * batch_size:(i + 1) *
                           batch_size, t_vec[:-1]].clone())
        out = torch.cat(out, axis=0)
        return out


def get_time_vector(size, length):
    return torch.linspace(0, 1, length).reshape(1, -1, 1).repeat(size, 1, 1)


def add_time(x, device):
    t = get_time_vector(x.shape[0], x.shape[1]).to(device)
    return torch.cat([x, t], dim=2)
