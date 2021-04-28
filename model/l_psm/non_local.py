'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
Description : 
LastEditTime: 2020-10-15 20:33:43
'''
import torch
import torch.nn as nn
import numpy as np


class NonLocal(nn.Module):
    def __init__(self, filters, spatial, temporal):
        super(NonLocal, self).__init__()

        self.filters = filters
        self.spatial = spatial
        self.temporal = temporal

        if self.spatial or self.temporal:
            self.non_local_conv_1 = nn.Linear(self.filters, self.filters)
            self.non_local_conv_2 = nn.Linear(self.filters, self.filters)
            self.non_local_conv_3 = nn.Linear(self.filters, self.filters)
            self.non_local_conv_4 = nn.Linear(self.filters * 2, 1)
        else:
            self.non_local_conv_1 = nn.Linear(self.filters, self.filters)
            self.non_local_conv_2 = nn.Linear(self.filters, self.filters)
            self.non_local_conv_3 = nn.Linear(self.filters, self.filters)
            self.non_local_conv_4 = nn.Linear(self.filters * 2, 2)

    def forward(self, x):
        x_shape = x.size()
        x_in = torch.Tensor.permute(x, (0, 2, 3, 1)).reshape(
            (x.shape[0], int(x.shape[2] * x.shape[3]), x.shape[1]))

        conv_1 = self.non_local_conv_1(x_in)
        conv_2 = self.non_local_conv_2(x_in)

        conv_2 = torch.Tensor.permute(conv_2, (0, 2, 1))
        matrix = conv_1.matmul(conv_2)
        matrix = nn.functional.softmax(matrix, dim=2)

        conv_3 = self.non_local_conv_3(x_in)
        conv_3 = matrix.matmul(conv_3)

        conv_3 = torch.cat((conv_3, x_in), dim=2).contiguous()
        offsets = self.non_local_conv_4(conv_3)

        offsets = torch.reshape(
            offsets, (x_shape[0], x_shape[2], x_shape[3], offsets.shape[-1])).permute((0, 3, 1, 2))
        return offsets


class ConvOffset2D_nonlocal2(nn.Module):
    def __init__(self, C_in, spatial=False, temporal=False, with_bn=True, bn_before_acfn=False, concat=True, if_self=True):
        super(ConvOffset2D_nonlocal2, self).__init__()
        self.C_in = C_in
        self.spatial = spatial
        self.temporal = temporal

        self.with_bn = with_bn
        self.bn_before_acfn = bn_before_acfn

        self.concat = concat
        self.self = if_self

        self.internal = nn.Conv2d(
            C_in, C_in, kernel_size=(1, 1), padding=(0, 0))
        self.internal_relu = nn.ReLU()

        self.external = nn.Conv2d(
            C_in, C_in, kernel_size=(1, 1), padding=(0, 0))
        self.external_relu = nn.ReLU()

        self.output = nn.Conv2d(C_in, C_in, kernel_size=(1, 1), padding=(0, 0))
        self.output_relu = nn.ReLU()

        if self.concat:
            self.fusion = nn.Conv2d(
                C_in * 2, 1, kernel_size=(1, 1), padding=(0, 0))

        if self.with_bn:
            self.internal_bn = nn.BatchNorm2d(C_in)
            self.external_bn = nn.BatchNorm2d(C_in)
            self.output_bn = nn.BatchNorm2d(C_in)

    def forward(self, x):
        '''
        Input: B, C, J, T
        Output: B, C, J, T, L
        '''
        B, C, J, T = x.shape

        x_int = self.internal(x)
        if self.with_bn and self.bn_before_acfn:
            x_int = self.internal_bn(x_int)
        x_int = self.internal_relu(x_int)
        if self.with_bn and not self.bn_before_acfn:
            x_int = self.internal_bn(x_int)

        x_ext = self.external(x)
        if self.with_bn and self.bn_before_acfn:
            x_ext = self.external_bn(x_ext)
        x_ext = self.external_relu(x_ext)
        if self.with_bn and not self.bn_before_acfn:
            x_ext = self.external_bn(x_ext)

        index = torch.zeros((B, J, T, 3), dtype=x.dtype, device=x.device)
        if self.concat:
            if self.spatial:
                x_int = torch.Tensor.permute(x_int, (0, 1, 3, 2)).unsqueeze(
                    dim=3).repeat((1, 1, 1, J, 1)).reshape((B, C, T, J * J))
                x_ext = torch.Tensor.permute(x_ext, (0, 1, 3, 2)).unsqueeze(
                    dim=4).repeat((1, 1, 1, 1, J)).reshape((B, C, T, J * J))

                x_concate = torch.cat((x_int, x_ext), dim=1)
                x_concate = self.fusion(x_concate)

                x_concate = torch.Tensor.reshape(
                    x_concate, (B, 1, T, J, J)).permute((0, 3, 2, 4, 1)).squeeze()

            elif self.temporal:
                x_int = x_int.unsqueeze(dim=3).repeat(
                    (1, 1, 1, T, 1)).reshape((B, C, J, T * T))
                x_ext = x_ext.unsqueeze(dim=4).repeat(
                    (1, 1, 1, 1, T)).reshape((B, C, J, T * T))

                x_concate = torch.cat((x_int, x_ext), dim=1)
                x_concate = self.fusion(x_concate)

                x_concate = torch.Tensor.reshape(
                    x_concate, (B, 1, J, T, T)).permute((0, 2, 3, 4, 1)).squeeze()

            else:
                raise RuntimeError

            matrix = x_concate
        else:
            x_int = torch.Tensor.permute(
                x_int, (0, 2, 3, 1)).reshape((B, J * T, C))  # x_int of shape (B, J*T, C), i.e. X^T
            # x_ext of shape (B, C, J*T), i.e. X
            x_ext = torch.Tensor.reshape(x_ext, (B, C, J * T))
            x_ext = torch.softmax(x_ext, dim=2)  # softmax(X)
            # softmax(X^T * X)

            matrix = x_int.matmul(x_ext).reshape(
                (B, J, T, J, T))  # X^T * softmax(X)
            if self.spatial:
                index = torch.arange(0, T, dtype=matrix.dtype, device=matrix.device).long().unsqueeze(dim=0)\
                    .unsqueeze(dim=1).unsqueeze(dim=3).unsqueeze(dim=4).repeat((B, J, 1, J, 1))

                matrix = torch.Tensor.gather(
                    matrix, index=index, dim=4).squeeze()  # Just need the spatial correlation of joints inside each frame

            elif self.temporal:
                index = torch.arange(0, J, dtype=matrix.dtype, device=matrix.device).long().unsqueeze(dim=0)\
                    .unsqueeze(dim=2).unsqueeze(dim=3).unsqueeze(dim=4).repeat((B, 1, T, T, 1))
                matrix = torch.Tensor.permute(matrix, (0, 1, 2, 4, 3))

                matrix = torch.Tensor.gather(
                    matrix, index=index, dim=4).squeeze()
            else:
                raise RuntimeError

        if self.spatial:  # compute the weighted sum
            output = torch.matmul(torch.Tensor.permute(
                matrix, (0, 2, 1, 3)), torch.Tensor.permute(x, (0, 3, 2, 1))).permute((0, 3, 2, 1))

        elif self.temporal:
            output = torch.matmul(matrix, torch.Tensor.permute(
                x, (0, 2, 3, 1))).permute((0, 3, 1, 2))
        else:
            raise RuntimeError

        if self.with_bn and self.bn_before_acfn:
            output = self.output_bn(output)
        output = self.output_relu(output)
        if self.with_bn and not self.bn_before_acfn:
            output = self.output_bn(output)

        if not self.self:
            index = torch.sort(torch.sort(matrix, dim=3, descending=True)[
                               1][:, :, :, :3], dim=3)[0]
            weight = torch.Tensor.gather(matrix, index=index, dim=3)
            index = index.unsqueeze(dim=1).repeat(
                (1, C, 1, 1, 1))  # B, C, J, T, 3
            weight = weight.unsqueeze(dim=1).repeat(
                (1, C, 1, 1, 1))  # B, C, J, T, 3

        else:
            if self.spatial:
                grid = torch.arange(0, J, dtype=matrix.dtype, device=matrix.device).unsqueeze(dim=0).\
                    unsqueeze(dim=0).unsqueeze(dim=3).repeat(
                        (B, T, 1, 1)).permute((0, 2, 1, 3))

            elif self.temporal:
                grid = torch.arange(0, T, dtype=matrix.dtype, device=matrix.device).unsqueeze(dim=0).\
                    unsqueeze(dim=0).unsqueeze(dim=3).repeat((B, J, 1, 1))

            if self.spatial:
                mask = torch.eye(J, dtype=x.dtype, device=x.device).unsqueeze(
                    0).unsqueeze(2).repeat((B, 1, T, 1)).byte()
            elif self.temporal:
                mask = torch.eye(T, dtype=x.dtype, device=x.device).unsqueeze(
                    0).unsqueeze(1).repeat((B, J, 1, 1)).byte()

            masked_matrix = torch.masked_fill(
                matrix, mask.bool(), value=-np.inf)  # mask the variance (i.e. self-correlation)
            # torch.sort(masked_matrix, dim=3, descending=True)[1][:, :, :, :2] is index of top 2 corr of each joint at each frame

            index = torch.sort(torch.cat((torch.sort(masked_matrix, dim=3, descending=True)[1][:, :, :, :2],
                                          grid.long()), dim=3), dim=3)[0]

            weight = torch.Tensor.gather(matrix, index=index, dim=3)
            index = index.unsqueeze(dim=1).repeat(
                (1, C, 1, 1, 1))  # B, C, J, T, 3
            weight = weight.unsqueeze(dim=1)

        if self.temporal:
            path = torch.Tensor.gather(x.unsqueeze(dim=4).repeat(
                (1, 1, 1, 1, 3)), index=index, dim=3)
            weighted = torch.Tensor.mul(path, weight)
        elif self.spatial:
            path = torch.Tensor.gather(x.unsqueeze(dim=4).repeat(
                (1, 1, 1, 1, 3)), index=index, dim=2)
            weighted = torch.Tensor.mul(path, weight)

        return output, path, index


class ConvOffset2D(nn.Module):
    """ConvOffset2D

    Convolutional layer responsible for learning the 2D offsets and output the
    deformed feature map using bilinear interpolation

    Note that this layer does not perform convolution on the deformed feature
    map. See get_deform_cnn in cnn.py 
    for usage
    """

    def __init__(self, filters, init_normal_stddev=0.01, non_local=False, spatial=True, temporal=True, kernel=3, padding=1, coords=None, **kwargs):
        """Init

        Parameters
        ----------
        filters : int
            Number of channel of the input feature map
        init_normal_stddev : float
            Normal kernel initialization
        **kwargs:
            Pass to superclass. See Con2d layer in pytorch
        """
        super(ConvOffset2D, self).__init__()
        self.filters = filters
        self.spatial = spatial
        self.temporal = temporal
        self.kernel = kernel
        self.padding = padding
        self.coords = coords

        self.non_local = non_local
        self._grid_param = None
        if self.spatial or self.temporal:
            if self.non_local:
                self.conv = NonLocal(
                    filters=self.filters, spatial=self.spatial, temporal=self.temporal)
            else:
                self.conv = nn.Conv2d(
                    self.filters, 1, self.kernel, padding=self.padding, bias=False, **kwargs)

        else:
            if self.non_local:
                self.conv = NonLocal(
                    filters=self.filters, spatial=self.spatial, temporal=self.temporal)
            else:
                self.conv = nn.Conv2d(
                    self.filters, 2, self.kernel, padding=self.padding, bias=False, **kwargs)

        if not self.non_local:
            self.conv.weight.data.copy_(self._init_weights(
                self.conv.weight, init_normal_stddev))

    def forward(self, x, coords=None):
        """Return the deformed featured map"""
        if coords is not None:
            self.coords = coords
        x_shape = x.size()
        # offsets = super(ConvOffset2D, self).forward(x)
        offsets = self.conv(x)
        offsets = torch.Tensor.unsqueeze(offsets, dim=1).repeat((1, x_shape[1], 1, 1, 1)).reshape((int(x_shape[0] * x_shape[1]), offsets.shape[1],
                                                                                                   x_shape[2], x_shape[3]))

        # offsets: (b*c, h, w, 2)
        # offsets = self._to_bc_h_w_2(offsets, x_shape)
        if self.spatial or self.temporal:
            offsets = self._to_bc_h_w_1_mine(
                offsets, x_shape)  # (256, 3, 11, 39, 2)
            y_offsets = torch.zeros(
                size=offsets.shape, dtype=offsets.dtype).to(x.device)
            if self.spatial:
                offsets = torch.stack([offsets, y_offsets], -1)
            elif self.temporal:
                offsets = torch.stack([y_offsets, offsets], -1)
            else:
                raise RuntimeError
        else:
            offsets = self._to_bc_h_w_2_mine(
                offsets, x_shape)  # (256, 3, 11, 39, 2)

        # x: (b*c, h, w)
        x = self._to_bc_h_w(x, x_shape)  # (256, 3, 11, 39)

        # X_offset: (b*c, h, w)
        x_offset, coords = th_batch_map_offsets(x, offsets, coords=self.coords, grid=self._get_grid(
            self, x), spatial=self.spatial, temporal=self.temporal)

        # x_offset: (b, h, w, c)
        x_offset = self._to_b_c_h_w(x_offset, x_shape)
        return x_offset, coords

    @staticmethod
    def _get_grid(self, x):
        batch_size, input_height, input_width = x.size(0), x.size(1), x.size(2)
        dtype, cuda = x.data.type(), x.data.is_cuda
        if self._grid_param == (batch_size, input_height, input_width, dtype, cuda):
            return self._grid
        self._grid_param = (batch_size, input_height, input_width, dtype, cuda)
        self._grid = th_generate_grid(
            batch_size, input_height, input_width, dtype, cuda, device=x.device)
        return self._grid

    @staticmethod
    def _init_weights(weights, std):
        fan_out = weights.size(0)
        fan_in = weights.size(1) * weights.size(2) * weights.size(3)
        w = np.random.normal(0.0, std, (fan_out, fan_in))
        return torch.from_numpy(w.reshape(weights.size()))

    @staticmethod
    def _to_bc_h_w_2(x, x_shape):
        """(b, 2c, h, w) -> (b*c, h, w, 2)"""
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]), 2)
        return x

    @staticmethod
    def _to_bc_h_w_2_mine(x, x_shape):
        """(b, 2c, h, w) -> (b*c, h, w, 2)"""
        # x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]), 2)
        x = x.contiguous().view(-1, 2, int(x_shape[2]), int(x_shape[3]))
        x = x.permute(0, 2, 3, 1).contiguous()

        return x

    @staticmethod
    def _to_bc_h_w_1_mine(x, x_shape):
        """(b, 2c, h, w) -> (b*c, h, w, 2)"""
        # x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]), 2)
        x = x.contiguous().view(-1, 1, int(x_shape[2]), int(x_shape[3]))
        x = x.permute(0, 2, 3, 1).contiguous()

        return x

    @staticmethod
    def _to_bc_h_w(x, x_shape):
        """(b, c, h, w) -> (b*c, h, w)"""
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))
        return x

    @staticmethod
    def _to_b_c_h_w(x, x_shape):
        """(b*c, h, w) -> (b, c, h, w)"""
        x = x.contiguous().view(-1,
                                int(x_shape[1]), int(x_shape[2]), int(x_shape[3]))
        return x


class ConvOffset2D_nonlocal(nn.Module):
    def __init__(self, C_in, spatial=False, temporal=False, with_bn=True, bn_before_acfn=False, concat=True, if_self=True):
        super(ConvOffset2D_nonlocal, self).__init__()
        self.C_in = C_in
        self.spatial = spatial
        self.temporal = temporal

        self.with_bn = with_bn
        self.bn_before_acfn = bn_before_acfn

        self.concat = concat
        self.self = if_self

        self.internal = nn.Conv2d(
            C_in, C_in, kernel_size=(1, 1), padding=(0, 0))
        self.internal_relu = nn.ReLU()

        self.external = nn.Conv2d(
            C_in, C_in, kernel_size=(1, 1), padding=(0, 0))
        self.external_relu = nn.ReLU()

        if self.concat:
            self.fusion = nn.Conv2d(
                C_in * 2, 1, kernel_size=(1, 1), padding=(0, 0))

        if self.with_bn:
            self.internal_bn = nn.BatchNorm2d(C_in)
            self.external_bn = nn.BatchNorm2d(C_in)

    def forward(self, x):
        '''
        Input: B, C, J, T
        Output: B, C, J, T, L
        '''
        B, C, J, T = x.shape

        x_int = self.internal(x)
        if self.with_bn and self.bn_before_acfn:
            x_int = self.internal_bn(x_int)
        x_int = self.internal_relu(x_int)
        if self.with_bn and not self.bn_before_acfn:
            x_int = self.internal_bn(x_int)

        x_ext = self.external(x)
        if self.with_bn and self.bn_before_acfn:
            x_ext = self.external_bn(x_ext)
        x_ext = self.external_relu(x_ext)
        if self.with_bn and not self.bn_before_acfn:
            x_ext = self.external_bn(x_ext)

        index = torch.zeros((B, J, T, 3), dtype=x.dtype, device=x.device)
        if self.concat:
            if self.spatial:
                x_int = torch.Tensor.permute(x_int, (0, 1, 3, 2)).unsqueeze(
                    dim=3).repeat((1, 1, 1, J, 1)).reshape((B, C, T, J * J))
                x_ext = torch.Tensor.permute(x_ext, (0, 1, 3, 2)).unsqueeze(
                    dim=4).repeat((1, 1, 1, 1, J)).reshape((B, C, T, J * J))

                x_concate = torch.cat((x_int, x_ext), dim=1)
                x_concate = self.fusion(x_concate)

                x_concate = torch.Tensor.reshape(
                    x_concate, (B, 1, T, J, J)).permute((0, 3, 2, 4, 1)).squeeze()

            elif self.temporal:
                x_int = x_int.unsqueeze(dim=3).repeat(
                    (1, 1, 1, T, 1)).reshape((B, C, J, T * T))
                x_ext = x_ext.unsqueeze(dim=4).repeat(
                    (1, 1, 1, 1, T)).reshape((B, C, J, T * T))

                x_concate = torch.cat((x_int, x_ext), dim=1)
                x_concate = self.fusion(x_concate)

                x_concate = torch.Tensor.reshape(
                    x_concate, (B, 1, J, T, T)).permute((0, 2, 3, 4, 1)).squeeze()

            else:
                raise RuntimeError

            matrix = x_concate
        else:
            x_int = torch.Tensor.permute(
                x_int, (0, 2, 3, 1)).reshape((B, J * T, C))
            x_ext = torch.softmax(x_ext, dim=2)
            x_ext = torch.Tensor.reshape(x_ext, (B, C, J * T))
            matrix = x_int.matmul(x_ext).reshape((B, J, T, J, T))

            if self.spatial:
                index = torch.arange(0, T, dtype=matrix.dtype, device=matrix.device).long().unsqueeze(dim=0)\
                    .unsqueeze(dim=1).unsqueeze(dim=3).unsqueeze(dim=4).repeat((B, J, 1, J, 1))

                matrix = torch.Tensor.gather(
                    matrix, index=index, dim=4).squeeze()

            elif self.temporal:
                index = torch.arange(0, J, dtype=matrix.dtype, device=matrix.device).long().unsqueeze(dim=0)\
                    .unsqueeze(dim=2).unsqueeze(dim=3).unsqueeze(dim=4).repeat((B, 1, T, T, 1))
                matrix = torch.Tensor.permute(matrix, (0, 1, 2, 4, 3))

                matrix = torch.Tensor.gather(
                    matrix, index=index, dim=4).squeeze()
            else:
                raise RuntimeError

        if not self.self:
            index = torch.sort(torch.sort(matrix, dim=3, descending=True)[
                               1][:, :, :, :3], dim=3)[0]
            weight = torch.Tensor.gather(matrix, index=index, dim=3)
            index = index.unsqueeze(dim=1).repeat(
                (1, C, 1, 1, 1))  # B, C, J, T, 3
            weight = weight.unsqueeze(dim=1).repeat(
                (1, C, 1, 1, 1))  # B, C, J, T, 3

        else:
            if self.spatial:
                grid = torch.arange(0, J, dtype=matrix.dtype, device=matrix.device).unsqueeze(dim=0).\
                    unsqueeze(dim=0).unsqueeze(dim=3).repeat(
                        (B, T, 1, 1)).permute((0, 2, 1, 3))

            elif self.temporal:
                grid = torch.arange(0, T, dtype=matrix.dtype, device=matrix.device).unsqueeze(dim=0).\
                    unsqueeze(dim=0).unsqueeze(dim=3).repeat((B, J, 1, 1))

            if self.spatial:
                mask = torch.eye(J, dtype=x.dtype, device=x.device).unsqueeze(
                    0).unsqueeze(2).repeat((B, 1, T, 1)).byte()
            elif self.temporal:
                mask = torch.eye(T, dtype=x.dtype, device=x.device).unsqueeze(
                    0).unsqueeze(1).repeat((B, J, 1, 1)).byte()
            masked_matrix = torch.masked_fill(
                matrix, mask.bool(), value=-np.inf)

            index = torch.sort(torch.cat((torch.sort(masked_matrix, dim=3, descending=True)[1][:, :, :, :2],
                                          grid.long()), dim=3), dim=3)[0]
            weight = torch.Tensor.gather(matrix, index=index, dim=3)
            index = index.unsqueeze(dim=1).repeat(
                (1, C, 1, 1, 1))  # B, C, J, T, 3
            weight = weight.unsqueeze(dim=1)

        if self.temporal:
            path = torch.Tensor.gather(x.unsqueeze(dim=4).repeat(
                (1, 1, 1, 1, 3)), index=index, dim=3)
            weighted = torch.Tensor.mul(path, weight)
        elif self.spatial:
            path = torch.Tensor.gather(x.unsqueeze(dim=4).repeat(
                (1, 1, 1, 1, 3)), index=index, dim=2)
            weighted = torch.Tensor.mul(path, weight)
        return weighted, path, index
