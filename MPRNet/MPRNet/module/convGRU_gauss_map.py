# import os
# import torch
# from torch import nn
from torch.autograd import Variable
#
#
# class ConvGRUCell(nn.Module):
#     def __init__(self, input_dim, hidden_dim, kernel_size, bias):
#         """
#         Initialize the ConvLSTM cell
#         :param input_size: (int, int)
#             Height and width of input tensor as (height, width).
#         :param input_dim: int
#             Number of channels of input tensor.
#         :param hidden_dim: int
#             Number of channels of hidden state.
#         :param kernel_size: (int, int)
#             Size of the convolutional kernel.
#         :param bias: bool
#             Whether or not to add the bias.
#         :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
#             Whether or not to use cuda.
#         """
#         super(ConvGRUCell, self).__init__()
#         self.padding = kernel_size[0] // 2, kernel_size[1] // 2
#         self.hidden_dim = hidden_dim
#         self.bias = bias
#
#         self.conv_gates = nn.Conv2d(in_channels=input_dim + hidden_dim,
#                                     out_channels=2*self.hidden_dim,  # for update_gate,reset_gate respectively
#                                     kernel_size=kernel_size,
#                                     padding=self.padding,
#                                     bias=self.bias)
#
#         self.conv_can = nn.Conv2d(in_channels=input_dim+hidden_dim,
#                               out_channels=self.hidden_dim, # for candidate neural memory
#                               kernel_size=kernel_size,
#                               padding=self.padding,
#                               bias=self.bias)
#
#     def init_hidden(self, batch_size):
#         return Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda()
#
#     def forward(self, input_tensor, h_cur):
#         """
#         :param self:
#         :param input_tensor: (b, c, h, w)
#             input is actually the target_model
#         :param h_cur: (b, c_hidden, h, w)
#             current hidden and cell states respectively
#         :return: h_next,
#             next hidden state
#         """
#         combined = torch.cat([input_tensor, h_cur], dim=1)
#         combined_conv = self.conv_gates(combined)
#
#         gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
#         reset_gate = torch.sigmoid(gamma)
#         update_gate = torch.sigmoid(beta)
#
#         combined = torch.cat([input_tensor, reset_gate*h_cur], dim=1)
#         cc_cnm = self.conv_can(combined)
#         cnm = torch.tanh(cc_cnm)
#
#         h_next = (1 - update_gate) * h_cur + update_gate * cnm
#         return h_next
#
#
# class ConvGRU(nn.Module):
#     def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
#                  dtype, batch_first=False, bias=True, return_all_layers=False):
#         """
#         :param input_size: (int, int)
#             Height and width of input tensor as (height, width).
#         :param input_dim: int e.g. 256
#             Number of channels of input tensor.
#         :param hidden_dim: int e.g. 1024
#             Number of channels of hidden state.
#         :param kernel_size: (int, int)
#             Size of the convolutional kernel.
#         :param num_layers: int
#             Number of ConvLSTM layers
#         :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
#             Whether or not to use cuda.
#         :param alexnet_path: str
#             pretrained alexnet parameters
#         :param batch_first: bool
#             if the first position of array is batch or not
#         :param bias: bool
#             Whether or not to add the bias.
#         :param return_all_layers: bool
#             if return hidden and cell states for all layers
#         """
#         super(ConvGRU, self).__init__()
#
#         # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
#         kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
#         hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
#         if not len(kernel_size) == len(hidden_dim) == num_layers:
#             raise ValueError('Inconsistent list length.')
#
#         self.height, self.width = input_size
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.kernel_size = kernel_size
#         self.dtype = dtype
#         self.num_layers = num_layers
#         self.batch_first = batch_first
#         self.bias = bias
#         self.return_all_layers = return_all_layers
#
#         cell_list = []
#         for i in range(0, self.num_layers):
#             cur_input_dim = input_dim if i == 0 else hidden_dim[i - 1]
#             cell_list.append(ConvGRUCell(input_size=(self.height, self.width),
#                                          input_dim=cur_input_dim,
#                                          hidden_dim=self.hidden_dim[i],
#                                          kernel_size=self.kernel_size[i],
#                                          bias=self.bias,
#                                          dtype=self.dtype))
#
#         # convert python list to pytorch module
#         self.cell_list = nn.ModuleList(cell_list)
#
#     def forward(self, input_tensor, hidden_state=None):
#         """
#         :param input_tensor: (b, t, c, h, w) or (t,b,c,h,w) depends on if batch first or not
#             extracted features from alexnet
#         :param hidden_state:
#         :return: layer_output_list, last_state_list
#         """
#         if not self.batch_first:
#             # (t, b, c, h, w) -> (b, t, c, h, w)
#             input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
#
#         # Implement stateful ConvLSTM
#         if hidden_state is not None:
#             raise NotImplementedError()
#         else:
#             hidden_state = self._init_hidden(batch_size=input_tensor.size(0))
#
#         layer_output_list = []
#         last_state_list   = []
#
#         seq_len = input_tensor.size(1)
#         cur_layer_input = input_tensor
#
#         for layer_idx in range(self.num_layers):
#             h = hidden_state[layer_idx]
#             output_inner = []
#             for t in range(seq_len):
#                 # input current hidden and cell state then compute the next hidden and cell state through ConvLSTMCell forward function
#                 h = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], # (b,t,c,h,w)
#                                               h_cur=h)
#                 output_inner.append(h)
#
#             layer_output = torch.stack(output_inner, dim=1)
#             cur_layer_input = layer_output
#
#             layer_output_list.append(layer_output)
#             last_state_list.append([h])
#
#         if not self.return_all_layers:
#             layer_output_list = layer_output_list[-1:]
#             last_state_list   = last_state_list[-1:]
#
#         return layer_output_list, last_state_list
#
#     def _init_hidden(self, batch_size):
#         init_states = []
#         for i in range(self.num_layers):
#             init_states.append(self.cell_list[i].init_hidden(batch_size))
#         return init_states
#
#     @staticmethod
#     def _check_kernel_size_consistency(kernel_size):
#         if not (isinstance(kernel_size, tuple) or
#                     (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
#             raise ValueError('`kernel_size` must be tuple or list of tuples')
#
#     @staticmethod
#     def _extend_for_multilayer(param, num_layers):
#         if not isinstance(param, list):
#             param = [param] * num_layers
#         return param
#
#
# if __name__ == '__main__':
#     # set CUDA device
#     os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#
#     # detect if CUDA is available or not
#     use_gpu = torch.cuda.is_available()
#     if use_gpu:
#         dtype = torch.cuda.FloatTensor # computation in GPU
#     else:
#         dtype = torch.FloatTensor
#
#     height = width = 6
#     channels = 256
#     hidden_dim = [32, 64]
#     kernel_size = (3,3) # kernel size for two stacked hidden layer
#     num_layers = 2 # number of stacked hidden layer
#     module = ConvGRU(input_size=(height, width),
#                     input_dim=channels,
#                     hidden_dim=hidden_dim,
#                     kernel_size=kernel_size,
#                     num_layers=num_layers,
#                     dtype=dtype,
#                     batch_first=True,
#                     bias = True,
#                     return_all_layers = False)
#
#     batch_size = 1
#     time_steps = 1
#     input_tensor = torch.rand(batch_size, time_steps, channels, height, width)  # (b,t,c,h,w)
#     layer_output_list, last_state_list = module(input_tensor)



import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import init


class ConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = 1
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

        # init.orthogonal(self.reset_gate.weight)
        # init.orthogonal(self.update_gate.weight)
        # init.orthogonal(self.out_gate.weight)
        # init.constant(self.reset_gate.bias, 0.)
        # init.constant(self.update_gate.bias, 0.)
        # init.constant(self.out_gate.bias, 0.)


    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        # batch_size = input_.data.size()[0]
        # spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided

        # data size is [batch, channel, height, width]
        # print(input_.shape)
        # print(prev_state.shape)
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state


class ConvGRU(nn.Module):

    def __init__(self, input_size, hidden_sizes, kernel_sizes, n_layers):
        '''
        Generates a multi-layer convolutional GRU.
        Preserves spatial dimensions across cells, only altering depth.
        Parameters
        ----------
        input_size : integer. depth dimension of input tensors.
        hidden_sizes : integer or list. depth dimensions of hidden state.
            if integer, the same hidden size is used for all cells.
        kernel_sizes : integer or list. sizes of Conv2d gate kernels.
            if integer, the same kernel size is used for all cells.
        n_layers : integer. number of chained `ConvGRUCell`.
        '''

        super(ConvGRU, self).__init__()

        self.input_size = input_size

        if type(hidden_sizes) != list:
            self.hidden_sizes = [hidden_sizes]*n_layers
        else:
            assert len(hidden_sizes) == n_layers, '`hidden_sizes` must have the same length as n_layers'
            self.hidden_sizes = hidden_sizes
        if type(kernel_sizes) != list:
            self.kernel_sizes = [kernel_sizes]*n_layers
        else:
            assert len(kernel_sizes) == n_layers, '`kernel_sizes` must have the same length as n_layers'
            self.kernel_sizes = kernel_sizes

        self.n_layers = n_layers

        # cells = []
        # for i in range(self.n_layers):
        #     if i == 0:
        #         input_dim = self.input_size
        #     else:
        #         input_dim = self.hidden_sizes[i-1]
        #
        #     self.cell = ConvGRUCell(input_dim, self.hidden_sizes[i], self.kernel_sizes[i])
        #     name = 'ConvGRUCell_' + str(i).zfill(2)
        #
        #     setattr(self, name, self.cell)
        #     cells.append(getattr(self, name))
        #
        # self.cells = cells
        self.cell = ConvGRUCell(512, 512, 3)


    def forward(self, x, hidden0):
        '''
        Parameters
        ----------
        x : 4D input tensor. (batch, channels, height, width).
        hidden : list of 4D hidden state representations. (batch, channels, height, width).
        Returns
        -------
        upd_hidden : 5D hidden representation. (layer, batch, channels, height, width).
        '''
        # if hidden is None:
        #     hidden = [None]*self.n_layers

        input_ = x

        # upd_hidden = []
        new_hidden0 = self.cell(input_, hidden0)
        # input_ = new_hidden0

        # for layer_idx in range(self.n_layers):
        #     cell = self.cells[layer_idx]
        #     if layer_idx == 0:
        #         new_hidden0 = cell(input_, hidden0)
        #         input_ = new_hidden0
        #     elif layer_idx == 1:
        #         new_hidden1 = cell(input_, hidden1)
        #         input_ = new_hidden1
        #     else:
        #         new_hidden2 = cell(input_, hidden2)
        #         input_ = new_hidden2

            # pass through layer
            # upd_cell_hidden = cell(input_, cell_hidden)
            # # upd_hidden.append(upd_cell_hidden)
            # # update input_ to the last updated hidden layer for next pass
            # input_ = upd_cell_hidden

        # retain tensors in list to allow different hidden sizes
        # print(upd_hidden[0].shape)
        return new_hidden0
