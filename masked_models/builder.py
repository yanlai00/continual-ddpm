import math
import torch
import torch.nn as nn
from masked_models.supsup_args import args
from . import modules
from . import init

class Builder(object):
    def __init__(self):
        self.linear_layer = modules.MultitaskMaskLinear
        self.conv_layer = modules.MultitaskMaskConv
        self.bn_layer = modules.MultitaskNonAffineBN
        self.conv_init = init.signed_constant
        self.transpose_conv_layer = modules.MultitaskMaskTransposeConv

    def activation(self):
        return nn.ReLU(inplace=True)

    def transposeconv(self,
                      in_planes,
                      out_planes,
                      kernel_size,
                      stride,
                      padding):
        transpose_conv = self.transpose_conv_layer(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
        self.conv_init(transpose_conv)
        return transpose_conv

    def conv(
        self,
        kernel_size,
        in_planes,
        out_planes,
        stride=1,
    ):

        if kernel_size == 1:
            conv = self.conv_layer(
                in_planes, out_planes, kernel_size=1, stride=stride, bias=False
            )
        elif kernel_size == 3:
            conv = self.conv_layer(
                in_planes,
                out_planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )
        elif kernel_size == 5:
            conv = self.conv_layer(
                in_planes,
                out_planes,
                kernel_size=5,
                stride=stride,
                padding=2,
                bias=False,
            )
        elif kernel_size == 7:
            conv = self.conv_layer(
                in_planes,
                out_planes,
                kernel_size=7,
                stride=stride,
                padding=3,
                bias=False,
            )
        else:
            return None

        self.conv_init(conv)
        return conv

    def conv1x1(
        self, in_planes, out_planes, stride=1
    ):
        """1x1 convolution with padding"""
        c = self.conv(
            1,
            in_planes,
            out_planes,
            stride=stride,
        )
        return c

    def conv3x3(
        self, in_planes, out_planes, stride=1
    ):
        """3x3 convolution with padding"""
        c = self.conv(
            3,
            in_planes,
            out_planes,
            stride=stride,
        )
        return c

    def conv5x5(
        self, in_planes, out_planes, stride=1
    ):
        """5x5 convolution with padding"""
        c = self.conv(
            5,
            in_planes,
            out_planes,
            stride=stride,
        )
        return c

    def conv7x7(
        self, in_planes, out_planes, stride=1
    ):
        """7x7 convolution with padding"""
        c = self.conv(
            7,
            in_planes,
            out_planes,
            stride=stride,
        )
        return c

    def nopad_conv5x5(
        self, in_planes, out_planes, stride=1,
    ):

        conv = self.conv_layer(
            in_planes,
            out_planes,
            kernel_size=5,
            stride=stride,
            padding=0,
            bias=False,
        )
        self.conv_init(conv)
        return conv

    def batchnorm(self, planes):
        return self.bn_layer(planes)

    def linear(self, in_channels, out_channels):
        lin = self.linear_layer(in_channels, out_channels)
        self.conv_init(lin)
        return lin