from __future__ import absolute_import, division

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
# from torch_deform_conv.deform_conv import th_batch_map_offsets, th_generate_grid

from scipy.ndimage.interpolation import map_coordinates as sp_map_coordinates



def th_flatten(a):
    """Flatten tensor"""
    return a.contiguous().view(a.nelement())


def th_repeat(a, repeats, axis=0):
    """Torch version of np.repeat for 1D"""
    assert len(a.size()) == 1
    return th_flatten(torch.transpose(a.repeat(repeats, 1), 0, 1))


def np_repeat_2d(a, repeats):
    """Tensorflow version of np.repeat for 2D"""

    assert len(a.shape) == 2
    a = np.expand_dims(a, 0)
    a = np.tile(a, [repeats, 1, 1])
    return a


def th_gather_2d(input, coords):
    inds = coords[:, 0]*input.size(1) + coords[:, 1]
    x = torch.index_select(th_flatten(input), 0, inds)
    return x.view(coords.size(0))


def th_map_coordinates(input, coords, order=1):
    """Tensorflow verion of scipy.ndimage.map_coordinates
    Note that coords is transposed and only 2D is supported
    Parameters
    ----------
    input : tf.Tensor. shape = (s, s)
    coords : tf.Tensor. shape = (n_points, 2)
    """

    assert order == 1
    input_size = input.size(0)

    coords = torch.clamp(coords, 0, input_size - 1)
    coords_lt = coords.floor().long()
    coords_rb = coords.ceil().long()
    coords_lb = torch.stack([coords_lt[:, 0], coords_rb[:, 1]], 1)
    coords_rt = torch.stack([coords_rb[:, 0], coords_lt[:, 1]], 1)

    vals_lt = th_gather_2d(input,  coords_lt.detach())
    vals_rb = th_gather_2d(input,  coords_rb.detach())
    vals_lb = th_gather_2d(input,  coords_lb.detach())
    vals_rt = th_gather_2d(input,  coords_rt.detach())

    coords_offset_lt = coords - coords_lt.type(coords.data.type())

    vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[:, 0]
    vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[:, 0]
    mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[:, 1]
    return mapped_vals


def sp_batch_map_coordinates(inputs, coords):
    """Reference implementation for batch_map_coordinates"""
    # coords = coords.clip(0, inputs.shape[1] - 1)

    assert (coords.shape[2] == 2)
    height = coords[:,:,0].clip(0, inputs.shape[1] - 1)
    width = coords[:,:,1].clip(0, inputs.shape[2] - 1)
    np.concatenate((np.expand_dims(height, axis=2), np.expand_dims(width, axis=2)), 2)

    mapped_vals = np.array([
        sp_map_coordinates(input, coord.T, mode='nearest', order=1)
        for input, coord in zip(inputs, coords)
    ])
    return mapped_vals


def th_batch_map_coordinates(input, coords, order=1):
    """Batch version of th_map_coordinates
    Only supports 2D feature maps
    Parameters
    ----------
    input : tf.Tensor. shape = (b, s, s)
    coords : tf.Tensor. shape = (b, n_points, 2)
    Returns
    -------
    tf.Tensor. shape = (b, s, s)
    """

    batch_size = input.size(0)
    input_height = input.size(1)
    input_width = input.size(2)

    n_coords = coords.size(1)

    # coords = torch.clamp(coords, 0, input_size - 1)

    coords = torch.cat((torch.clamp(coords.narrow(2, 0, 1), 0, input_height - 1), torch.clamp(coords.narrow(2, 1, 1), 0, input_width - 1)), 2)

    assert (coords.size(1) == n_coords)

    coords_lt = coords.floor().long()
    coords_rb = coords.ceil().long()
    coords_lb = torch.stack([coords_lt[..., 0], coords_rb[..., 1]], 2)
    coords_rt = torch.stack([coords_rb[..., 0], coords_lt[..., 1]], 2)
    idx = th_repeat(torch.arange(0, batch_size), n_coords).long()
    idx = Variable(idx, requires_grad=False)
    if input.is_cuda:
        idx = idx.to(input.device)

    def _get_vals_by_coords(input, coords):
        indices = torch.stack([
            idx, th_flatten(coords[..., 0]), th_flatten(coords[..., 1])
        ], 1)
        inds = indices[:, 0]*input.size(1)*input.size(2)+ indices[:, 1]*input.size(2) + indices[:, 2]
        
        vals = th_flatten(input).index_select(0, inds)
        vals = vals.view(batch_size, n_coords)
        return vals

    vals_lt = _get_vals_by_coords(input, coords_lt.detach())
    vals_rb = _get_vals_by_coords(input, coords_rb.detach())
    vals_lb = _get_vals_by_coords(input, coords_lb.detach())
    vals_rt = _get_vals_by_coords(input, coords_rt.detach())

    coords_offset_lt = coords - coords_lt.type(coords.data.type())
    vals_t = coords_offset_lt[..., 0]*(vals_rt - vals_lt) + vals_lt
    vals_b = coords_offset_lt[..., 0]*(vals_rb - vals_lb) + vals_lb
    mapped_vals = coords_offset_lt[..., 1]* (vals_b - vals_t) + vals_t
    
    # print(mapped_vals.shape)
    return mapped_vals


def sp_batch_map_offsets(input, offsets):
    """Reference implementation for tf_batch_map_offsets"""

    batch_size = input.shape[0]
    input_height = input.shape[1]
    input_width = input.shape[2]

    offsets = offsets.reshape(batch_size, -1, 2)
    grid = np.stack(np.mgrid[:input_height, :input_width], -1).reshape(-1, 2)
    grid = np.repeat([grid], batch_size, axis=0)
    coords = offsets + grid
    # coords = coords.clip(0, input_size - 1)

    mapped_vals = sp_batch_map_coordinates(input, coords)
    return mapped_vals


def th_generate_grid(batch_size, input_height, input_width, dtype, cuda, device):
    grid = np.meshgrid(
        range(input_height), range(input_width), indexing='ij'
    )
    # grid: 2 * (height * width)

    grid = np.stack(grid, axis=-1)
    grid = grid.reshape(-1, 2)
    # grid: (height * width) * 2

    grid = np_repeat_2d(grid, batch_size)
    grid = torch.from_numpy(grid).type(dtype)
    if cuda:
        grid = grid.to(device)
    return Variable(grid, requires_grad=False)


def th_batch_map_offsets(input, offsets, grid=None, order=1, spatial=False, temporal=False, coords=None):
    """Batch map offsets into input
    Parameters
    ---------
    input : torch.Tensor. shape = (b, s, s)      -> (b * c, w, h)
    offsets: torch.Tensor. shape = (b, s, s, 2)       -> (b * p, w, h, 2)
    Returns
    -------
    torch.Tensor. shape = (b, s, s)     -> (b, p, c, w, h)
    """
    batch_size = input.size(0)
    input_height = input.size(1)
    input_width = input.size(2)

    offsets = offsets.view(batch_size, -1, 2)


    if grid is None:
        grid = th_generate_grid(batch_size, input_height, input_width, offsets_1.data.type(), offsets_1.data.is_cuda, device=input.device)
    # grid -> (b * p, w * h, 2)

    # offset_tmp = offsets.data.cpu().numpy()
    if spatial or temporal:
        if spatial:
            offsets[:, :, 0] = (torch.sigmoid(offsets[:, :, 0]) - 0.5) * input_height * 2
        if temporal:
            offsets[:, :, 1] = torch.sigmoid(offsets[:, :, 1]) * input_width
    else:
        offsets[:, :, 0] = (torch.sigmoid(offsets[:, :, 0]) - 0.5) * input_height * 2
        offsets[:, :, 1] = torch.sigmoid(offsets[:, :, 1]) * input_width


    if coords is not None:
        coords = offsets + coords
    else:
        coords = offsets + grid

    mapped_vals = th_batch_map_coordinates(input, coords)
    return mapped_vals, coords







class ConvOffset2D_multi(nn.Module):
    """ConvOffset2D

    Convolutional layer responsible for learning the 2D offsets and output the
    deformed feature map using bilinear interpolation

    Note that this layer does not perform convolution on the deformed feature
    map. See get_deform_cnn in cnn.py for usage
    """
    def __init__(self, filters, init_normal_stddev=0.01, spatial=False, temporal=False, non_local=False, **kwargs):
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
        super(ConvOffset2D_multi, self).__init__()
        self.filters = filters
        self.spatial = True
        self.temporal = False

        self.kernel = 3
        self.padding = 1
        self.non_local = False
        self._grid_param = None

        self.conv_1 = ConvOffset2D(filters=self.filters, init_normal_stddev=init_normal_stddev, spatial=self.spatial, \
            temporal=self.temporal, kernel=self.kernel, padding=self.padding, non_local=self.non_local)
        self.conv_2 = ConvOffset2D(filters=self.filters, init_normal_stddev=init_normal_stddev, spatial=self.spatial, \
            temporal=self.temporal, kernel=self.kernel, padding=self.padding, non_local=self.non_local)
        self.conv_3 = ConvOffset2D(filters=self.filters, init_normal_stddev=init_normal_stddev, spatial=self.spatial, \
            temporal=self.temporal, kernel=self.kernel, padding=self.padding, non_local=self.non_local)


    def forward(self, x):
        """Return the deformed featured map"""

        mapped_vals_1, coords_1 = self.conv_1(x)
        mapped_vals_2, coords_2 = self.conv_2(x, coords_1)


        mapped_vals = torch.stack([x.unsqueeze(dim=4), mapped_vals_1.unsqueeze(dim=4), mapped_vals_2.unsqueeze(dim=4)], dim=4)
        coords = torch.stack([torch.zeros(coords_1.shape, dtype=coords_1.dtype, device=coords_1.device).unsqueeze(dim=3), coords_1.unsqueeze(dim=3), \
            coords_2.unsqueeze(dim=3)], dim=3)


        coords = torch.Tensor.reshape(coords, (x.shape[0], x.shape[1], x.shape[2], x.shape[3], 2, 3))
        mapped_vals = mapped_vals.squeeze()
        return mapped_vals, coords

    @staticmethod
    def _get_grid(self, x):
        batch_size, input_height, input_width = x.size(0), x.size(1), x.size(2)
        dtype, cuda = x.data.type(), x.data.is_cuda
        if self._grid_param == (batch_size, input_height, input_width, dtype, cuda):
            return self._grid
        self._grid_param = (batch_size, input_height, input_width, dtype, cuda)
        self._grid = th_generate_grid(batch_size, input_height, input_width, dtype, cuda, device=x.device)
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
        # x = x.contiguous().view(-1, int(x_shape[1]), int(x_shape[2]), int(x_shape[3]))
        x = x.reshape((int(x.shape[0] / x_shape[1]), x_shape[1], x_shape[2], x_shape[3], 3))
        return x



            


