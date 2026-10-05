#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Template: wrappers that HOLD nn.Linear / nn.Conv2d as member variables
(composition, not inheritance).  The constructor arguments are the same as
the originals, and the forward pass just delegates for now.

Put your own processing at the places marked  # >>> HOOK.
"""

import torch
import torch.nn as nn


class WiGLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, n_chunks=1):
        super().__init__()
        self._n_chunks = n_chunks
        self._out_features = out_features * n_chunks
        self.linear = nn.Linear(in_features, self._out_features * 2, bias=bias,
                                device=device, dtype=dtype)
        self.reset_gate_parameters()

    @torch.no_grad()
    def reset_gate_parameters(self):
        # rows [out_features:] of the weight belong to the gate g
        self.linear.weight[self._out_features:].zero_()

    def forward(self, x):
        y, g = self.linear(x).chunk(2, dim=-1)
        y = y * torch.sigmoid(g)
        if( self._n_chunks > 1 ):
            y = sum(y.chunk( self._n_chunks, dim=-1 )) / self._n_chunks
        return y

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    @property
    def in_features(self):
        return self.linear.in_features

    @property
    def out_features(self):
        return self._out_features // self._n._n_chunks


class WiGConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, bias=True, padding_mode='zeros',
                 device=None, dtype=None, n_chunks=1):
        super().__init__()
        self._n_chunks = n_chunks
        self._out_channels = out_channels * n_chunks
        self.conv = nn.Conv2d(in_channels, self._out_channels * 2, kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              bias=bias, padding_mode=padding_mode,
                              device=device, dtype=dtype)
        self.reset_gate_parameters()

    @torch.no_grad()
    def reset_gate_parameters(self):
        self.conv.weight[self._out_channels:].zero_()

    def forward(self, x):
        y, g = self.conv(x).chunk(2, dim=1)
        y = y * torch.sigmoid(g)
        if( self._n_chunks > 1 ):
            y = sum(y.chunk( self._n_chunks, dim=1 )) / self._n_chunks
        return y

    @property
    def weight(self):
        return self.conv.weight

    @property
    def bias(self):
        return self.conv.bias

    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self._out_channels // self._n._n_chunks

    @property
    def kernel_size(self):
        return self.conv.kernel_size


if __name__ == '__main__':
    lin = WiGLinear(16, 8)
    print(lin(torch.randn(4, 16)).shape)            # (4, 8)
    lin = WiGLinear(16, 8, n_chunks=4)
    print(lin(torch.randn(4, 16)).shape)            # (4, 8)

    conv = WiGConv2d(3, 8, 3, padding=1)
    print(conv(torch.randn(4, 3, 32, 32)).shape)    # (4, 8, 32, 32)
    conv = WiGConv2d(3, 8, 3, padding=1, n_chunks=4)
    print(conv(torch.randn(4, 3, 32, 32)).shape)    # (4, 8, 32, 32)

