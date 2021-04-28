'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
Description : 
LastEditTime: 2020-10-16 02:29:44
'''
import torch
import torch.nn as nn
import signatory
import numpy as np

ntu_spatial_path = np.array([    [17, 1, 13],    [21, 2, 1],    [4, 3, 21],     [3, 4, 3],    [21, 5, 6],    [5, 6, 7],    [6, 7, 8],    [7, 8, 7],    [10, 9, 23],    [11, 10, 9],    [12, 11, 10],     [25, 12, 11],    [1, 13, 14],    [13, 14, 15],     [14, 15, 16],    [15, 16, 15],    [18, 17, 1],    [19, 18, 17],    [20, 19, 18],    [19, 20, 19],    [9, 21, 5],    [23, 22, 23],    [8, 23, 22],    [25, 24, 25],    [24, 25, 12]]).astype(np.float32)
ntu_spatial_path -= 1

chalearn13_hand_crafted_spatial_path = np.array([
    [1, 0, 1], 
    [0, 1, 2],
    [1, 2, 1],
    [4, 3, 1], 
    [5, 4, 3],
    [6, 5, 4],
    [5, 6, 5],
    [8, 7, 1],
    [9, 8, 7],
    [10, 9, 8],
    [9, 10, 9]
]).astype(np.float32)


class SigNet(nn.Module):
    def __init__(self, in_channels, out_dimension, sig_depth, include_time=True):
        super(SigNet, self).__init__()
        self.augment = signatory.Augment(in_channels=in_channels,
                                         layer_sizes=(),
                                         kernel_size=1,
                                         include_original=True,
                                         include_time=include_time)
        self.signature = signatory.Signature(depth=sig_depth)

        if include_time:
            # +1 because signatory.Augment is used to add time as well
            sig_channels = signatory.signature_channels(channels=in_channels + 1,
                                                        depth=sig_depth)
        else:
            sig_channels = signatory.signature_channels(channels=in_channels,
                                                        depth=sig_depth)
        # pdb.set_trace()
        self.linear = torch.nn.Linear(sig_channels,
                                      out_dimension)

    def forward(self, inp):
        # inp is a three dimensional tensor of shape (batch, stream, in_channels)
        x = self.augment(inp)
        # pdb.set_trace()

        if x.size(1) <= 1:
            raise RuntimeError("Given an input with too short a stream to take the"
                               " signature")
        # x in a three dimensional tensor of shape (batch, stream, in_channels + 1),
        # as time has been added as a value
        y = self.signature(x, basepoint=True)
        # pdb.set_trace()

        # y is a two dimensional tensor of shape (batch, terms), corresponding to
        # the terms of the signature
        z = self.linear(y)
        # z is a two dimensional tensor of shape (batch, out_dimension)

        # pdb.set_trace()

        return z


class SigModule(nn.Module):
    def __init__(self, in_channels, sig_in_channels, out_dimension, sig_depth, \
        win_size=5, use_bottleneck=False):

        super(SigModule, self).__init__()

        self.in_channels = in_channels
        self.sig_in_channels = sig_in_channels
        self.out_dimension = out_dimension
        self.sig_depth = sig_depth
        self.win_size = win_size
        self.use_bottleneck = use_bottleneck

        if self.use_bottleneck:
            self.bottleneck = nn.Conv2d(self.in_channels, self.sig_in_channels, kernel_size=(1, 1))
            self.relu = nn.ReLU()
            self.bn = nn.BatchNorm2d(self.sig_in_channels)

        self.sig = SigNet(self.sig_in_channels, self.out_dimension, self.sig_depth)


    def forward(self, x):

        # start = time.time()

        if self.use_bottleneck:
            x = self.bn(self.relu(self.bottleneck(x)))

        N, C, J, T = x.shape
        delta_t = (self.win_size - 1) // 2

        floor_t = 0
        ceil_t = T

        self.res = torch.zeros([N, self.out_dimension, J, T], dtype=torch.float, requires_grad=True).to(x.device)

        for i in range(J):
            for j in range(T):

                start_t = np.clip(j - delta_t, floor_t, ceil_t)
                end_t = np.clip(j + delta_t + 1, floor_t, ceil_t)

                stream_cur = x[:, :, i, start_t: end_t]
                stream_cur = stream_cur.permute(0, 2, 1)    # NCT to NTC

                feat_cur = self.sig(stream_cur)

                self.res[:, :, i, j] = feat_cur

        return self.res


class SigModuleParallel(nn.Module):
    def __init__(self, in_channels, sig_in_channels, out_dimension, sig_depth, \
        win_size=5, use_bottleneck=False, specific_path=None, spatial_ps=False):

        """
            - inputs:
                specific_path: list(tuple). Specify the path. [(y1, x1), (y2, x2), ...]
        """

        super(SigModuleParallel, self).__init__()

        self.in_channels = in_channels
        self.sig_in_channels = sig_in_channels
        self.out_dimension = out_dimension
        self.sig_depth = sig_depth
        self.win_size = win_size
        self.use_bottleneck = use_bottleneck
        self.specific_path = specific_path
        self.spatial_ps = spatial_ps

        # pdb.set_trace()

        assert self.win_size % 2 == 1

        if self.use_bottleneck:
            self.bottleneck = nn.Conv2d(self.in_channels, self.sig_in_channels, kernel_size=(1, 1))
            self.relu = nn.ReLU()
            self.bn = nn.BatchNorm2d(self.sig_in_channels)


        self.delta_t = (self.win_size - 1) // 2

        if self.specific_path is None:
            self.pad = nn.ReplicationPad2d((self.delta_t, self.delta_t, 0, 0))
        else:
            self.pad = nn.ReplicationPad2d((self.delta_t, self.delta_t, self.delta_t, self.delta_t))
        # pdb.set_trace()
        self.sig = SigNet(self.sig_in_channels, self.out_dimension, self.sig_depth, include_time=False if self.spatial_ps else True)


    def forward(self, x):

        # start = time.time()

        if self.use_bottleneck:
            x = self.bn(self.relu(self.bottleneck(x)))

        N, C, J, T = x.shape

        x = self.pad(x)

        if self.specific_path is None:
            self.input = torch.zeros([J*T*N, self.win_size, C], dtype=torch.float, requires_grad=True).to(x.device)
        else:
            self.input = torch.zeros([J*T*N, len(self.specific_path), C], dtype=torch.float, requires_grad=True).to(x.device)

        self.res = torch.zeros([N, self.out_dimension, J, T], dtype=torch.float, requires_grad=True).to(x.device)


        if self.specific_path is None:
            for i in range(J):
                for j in range(T):
                    self.input[(i*T+j)*N:(i*T+j+1)*N, :, :] = x[:, :, i, j:j+self.win_size].permute(0, 2, 1)
        else:
            for i in range(self.delta_t, self.delta_t+J):
                for j in range(self.delta_t, self.delta_t+T):
                    for k, coords in enumerate(self.specific_path):
                        self.input[((i-self.delta_t)*T+(j-self.delta_t))*N:((i-self.delta_t)*T+(j-self.delta_t)+1)*N, k, :] = \
                            x[:, :, i+coords[0]-self.delta_t, j+coords[1]-self.delta_t]


        feat_cur = self.sig(self.input)

        for i in range(J):
            for j in range(T):
                self.res[:, :, i, j] = feat_cur[(i*T+j)*N:(i*T+j+1)*N, :]

        return self.res


class SigModuleParallel_cheng(nn.Module):
    def __init__(self, in_channels, sig_in_channels, out_dimension, sig_depth, \
        win_size=5, use_bottleneck=False, specific_path=None, spatial_ps=False):

        """
            - inputs:
                specific_path: list(tuple). Specify the path. [(y1, x1), (y2, x2), ...]
        """

        super(SigModuleParallel_cheng, self).__init__()

        self.in_channels = in_channels
        self.sig_in_channels = sig_in_channels
        self.out_dimension = out_dimension
        self.sig_depth = sig_depth
        self.win_size = win_size
        self.use_bottleneck = use_bottleneck
        self.specific_path = specific_path
        self.spatial_ps = spatial_ps

        assert self.win_size % 2 == 1

        if self.use_bottleneck:
            self.bottleneck = nn.Conv2d(self.in_channels, self.sig_in_channels, kernel_size=(1, 1))
            self.relu = nn.ReLU()
            self.bn = nn.BatchNorm2d(self.sig_in_channels)


        self.delta_t = (self.win_size - 1) // 2

        if self.specific_path is None:
            self.pad = nn.ReplicationPad2d((self.delta_t, self.delta_t, 0, 0))
        else:
            self.pad = nn.ReplicationPad2d((self.delta_t, self.delta_t, self.delta_t, self.delta_t))
        self.sig = SigNet(self.sig_in_channels, self.out_dimension, self.sig_depth, include_time=False if self.spatial_ps else True)


    def forward(self, x):

        N, C, J, T, L = x.shape

        if self.use_bottleneck:
            x = torch.Tensor.permute(x, (0, 4, 1, 2, 3)).reshape((N * L, C, J, T))
            x = self.bn(self.relu(self.bottleneck(x)))
            x = torch.Tensor.reshape(x, (N, L, x.shape[1], J, T)).permute((0, 2, 3, 4, 1))

        if self.specific_path is None:
            self.input = torch.zeros([J*T*N, self.win_size, self.sig_in_channels], dtype=torch.float, requires_grad=True).to(x.device)
        else:
            self.input = torch.zeros([J*T*N, len(self.specific_path), C], dtype=torch.float, requires_grad=True).to(x.device)

        self.res = torch.zeros([N, self.out_dimension, J, T], dtype=torch.float, requires_grad=True).to(x.device)


        if self.specific_path is None:
            for i in range(J):
                for j in range(T):
                    self.input[(i * T + j) * N:(i * T  + j + 1) * N, :, :] = x[:, :, i, j, :].permute(0, 2, 1)
        else:
            for i in range(self.delta_t, self.delta_t+J):
                for j in range(self.delta_t, self.delta_t+T):
                    for k, coords in enumerate(self.specific_path):
                        self.input[((i-self.delta_t)*T+(j-self.delta_t))*N:((i-self.delta_t)*T+(j-self.delta_t)+1)*N, k, :] = \
                            x[:, :, i+coords[0]-self.delta_t, j+coords[1]-self.delta_t]


        feat_cur = self.sig(self.input)
        for i in range(J):
            for j in range(T):

                self.res[:, :, i, j] = feat_cur[(i*T+j)*N:(i*T+j+1)*N, :]
        return self.res


class STEM(nn.Module):
    def __init__(self, spatial, temporal, conv, C_in, C_out, with_bn=True, bn_before_actfn=False):
        super(STEM, self).__init__()
        self.spatial = spatial
        self.temporal = temporal
        self.conv = conv

        self.with_bn = with_bn
        self.bn_before_actfn = bn_before_actfn

        self.C_in = C_in
        self.C_out = C_out

        if self.spatial or self.temporal:
            if self.conv:
                self.fusion = nn.Sequential()
                self.fusion.add_module("conv", nn.Conv2d(int(C_out * 2), C_out, kernel_size=(1, 1), padding=(0, 0)))
                if self.with_bn and self.bn_before_actfn:
                    self.fusion.add_module("bn", nn.BatchNorm2d(C_out))
                self.fusion.add_module("relu", nn.ReLU())
                if self.with_bn and not self.bn_before_actfn:
                    self.fusion.add_module("bn", nn.BatchNorm2d(C_out))

            if self.spatial and self.temporal:
                self.ps_fusion = nn.Sequential()
                self.ps_fusion.add_module("conv", nn.Conv2d(int(C_out * 2), C_out, kernel_size=(1, 1), padding=(0, 0)))
                if self.with_bn and self.bn_before_actfn:
                    self.ps_fusion.add_module("bn", nn.BatchNorm2d(C_out))
                self.ps_fusion.add_module("relu", nn.ReLU())
                if self.with_bn and not self.bn_before_actfn:
                    self.ps_fusion.add_module("bn", nn.BatchNorm2d(C_out))


            if self.spatial:
                self.spatial_sig = SigNet(C_in, C_out, sig_depth=3, include_time=True)
            if self.temporal:
                self.temporal_sig = SigNet(C_in, C_out, sig_depth=3, include_time=True)
        
        if self.conv:
            self.raw = nn.Sequential()
            self.raw.add_module("conv", nn.Conv2d(C_in, C_out, kernel_size=(1, 3), padding=(0, 1)))
            if self.with_bn and self.bn_before_actfn:
                self.raw.add_module("bn", nn.BatchNorm2d(C_out))
            self.raw.add_module("relu", nn.ReLU())
            if self.with_bn and not self.bn_before_actfn:
                self.raw.add_module("bn", nn.BatchNorm2d(C_out))

    
    def forward(self, x):
        B, C, J, T = x.shape
        if self.conv:
            x_conv = self.raw(x)
        
        if not self.spatial and not self.temporal:
            return x_conv

        if self.spatial:
            spatial_path = torch.zeros((int(B * J * T), 3, C), dtype=x.dtype, device=x.device)
            for i in range(J):
                spatial_path[B * T * i: B * T * (i + 1), :, :] = torch.index_select(x, dim=2, \
                    index=torch.from_numpy(ntu_spatial_path[i, :].ravel()).long().to(x.device)).permute((0, 3, 2, 1)).reshape((B * T, 3, C))
            spatial_path = spatial_path.contiguous()
            
            spatial_sig = self.spatial_sig(spatial_path)
            spatial_sig = torch.Tensor.reshape(spatial_sig, (J, B, T, self.C_out)).permute((1, 3, 0, 2))

        if self.temporal:
            temporal_path = torch.zeros((int(B * J * T), 3, C), dtype=x.dtype, device=x.device)
            x_ps = torch.nn.functional.pad(x, (1, 1, 0, 0), 'replicate')
            for i in range(T):
                temporal_path[(B * J) * i:(B * J) * (i + 1), :, :] = x_ps[:, :, :, i:i + 3].permute((0, 2, 3, 1)).reshape((B * J, 3, C))
            temporal_path = temporal_path.contiguous()
            
            temporal_sig = self.temporal_sig(temporal_path)
            temporal_sig = torch.Tensor.reshape(temporal_sig, (T, B, J, self.C_out)).permute((1, 3, 2, 0))
        
        
        if self.spatial and self.temporal:
            sig = torch.cat((spatial_sig, temporal_sig), dim=1)
            sig = self.ps_fusion(sig)

        elif self.spatial:
            sig = spatial_sig
        elif self.temporal:
            sig = temporal_sig
        if not self.conv:
            return sig

        x_out = torch.cat((x_conv, sig), dim=1)
        x_out = self.fusion(x_out)

        return x_out