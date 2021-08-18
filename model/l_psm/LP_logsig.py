import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata
import signatory

from utils import init_weights
from model.l_psm.resfnn import ResFNN as ResFNN
from model.l_psm.ffn import FFN


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
        super(sp, self).__init__()

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


def compute_multilevel_logsignature(driving_path: torch.Tensor, time_driving: torch.Tensor, time_u: torch.Tensor, time_t: torch.Tensor, depth: int):
    """

    Parameters
    ----------
    driving_path: torch.Tensor
        Tensor of shape [batch_size, L, dim] where L is big enough so that we consider this 
    time_driving: torch.Tensor
        Time evaluations of driving_path
    time_u: torch.Tensor
        Time discretisation used to calculate logsignatures
    time_t: torch.Tensor
        Time discretisation of generated path
    depth: int
        depth of logsignature

    Returns
    -------
    multi_level_signature: torch.Tensor

    ind_u: List
        List of indices time_u used in the logsigrnn
    """
    logsig_channels = signatory.logsignature_channels(
        in_channels=driving_path.shape[-1], depth=depth)

    multi_level_log_sig = []

    u_logsigrnn = []
    last_u = -1
    start_points = []
    for ind_t, t in enumerate(time_t):
        u = time_u[time_u <= t].max()
        ind_low = torch.nonzero(
            (time_driving <= u).float(), as_tuple=False).max()
        if u != last_u:
            u_logsigrnn.append(u)
            last_u = u

        ind_max = torch.nonzero(
            (time_driving <= t).float(), as_tuple=False).max()
        interval = driving_path[:, ind_low:ind_max + 1, :]
        # if t == 0:
        multi_level_log_sig.append(signatory.logsignature(
            interval, depth=depth, basepoint=True))
        start_points.append(driving_path[:, ind_low, :])
        # else:
        #    multi_level_log_sig[:,ind_t] = signatory.logsignature(interval, depth=depth)

    return multi_level_log_sig, start_points, u_logsigrnn


class LogSigRNN(nn.Module):
    def __init__(self, input_dim, output_dim, n_lags, depth, hidden_dim, len_interval_u):

        super(LogSigRNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        input_dim_rnn = input_dim

        logsig_channels = signatory.logsignature_channels(
            in_channels=input_dim_rnn, depth=depth)

        self.depth = depth
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.len_interval_u = len_interval_u
        # self.time_t = torch.linspace(0,1,n_lags)

        # definition of LSTM + linear at the end
        self.rnn = nn.Sequential(
            FFN(input_dim=hidden_dim + logsig_channels + input_dim,
                output_dim=hidden_dim,
                hidden_dims=[hidden_dim, hidden_dim]),
            nn.Tanh()
        )
        self.linear = nn.Linear(hidden_dim, output_dim, bias=False)

        self.rnn.apply(init_weights)
        self.linear.apply(init_weights)

        # neural network to initialise h0 from the LSTM
        self.initial_nn = nn.Sequential(
            ResFNN(input_dim, hidden_dim, [hidden_dim, hidden_dim]), nn.Tanh())
        self.initial_nn.apply(init_weights)

    def forward(self, x, n_lags: int, device: str,):
        batch_size = x.size(0)
        time_t = torch.linspace(0, 1, n_lags).to(device)
        nT = x.size(1)
        time_driving = torch.linspace(0, 1, nT)
        time_u = time_driving[::self.len_interval_u]
        x_logsig, start_points, u_logsigrnn = compute_multilevel_logsignature(driving_path=x, time_driving=time_driving.to(
            device), time_u=time_u.to(device), time_t=time_t.to(device), depth=self.depth)
        u_logsigrnn.append(time_t[-1])

        last_h = self.initial_nn(x[:, 0, :])
        output = torch.zeros(batch_size, n_lags,
                             self.output_dim, device=device)
        for idx, (t, start_points_, x_logsig_) in enumerate(zip(time_t, start_points, x_logsig)):
            h = self.rnn(torch.cat([last_h, start_points_, x_logsig_], -1))
            if t >= u_logsigrnn[0]:
                del u_logsigrnn[0]
                last_h = h
            output[:, idx, :] = self.linear(h)

        assert output.shape[1] == n_lags
        return output
