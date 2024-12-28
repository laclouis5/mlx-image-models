import math
from typing import Callable

import mlx.core as mx
import numpy as np
from mlx import nn

from .conv2d_same import conv2d_same
from .helpers import _int_tuple_2_t, to_2tuple
from .padding import get_padding_value


# expert_shape = (O, K, K, I // G)
def get_condconv_initializer(
    initializer: Callable[[mx.array], mx.array], num_experts: int, expert_shape
):
    # (E, P)
    def condconv_initializer(weight: mx.array):
        """CondConv initializer function."""
        num_params = np.prod(expert_shape)
        if (
            len(weight.shape) != 2
            or weight.shape[0] != num_experts
            or weight.shape[1] != num_params
        ):
            raise (
                ValueError(
                    "CondConv variables must have shape [num_experts, num_params]"
                )
            )
        weight[:] = initializer(weight)

    return condconv_initializer


class CondConv2d(nn.Module):
    """Conditionally Parameterized Convolution
    Inspired by: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/condconv/condconv_layers.py

    Grouped convolution hackery for parallel execution of the per-sample kernel filters inspired by this discussion:
    https://github.com/pytorch/pytorch/issues/17983
    """

    __constants__ = ["in_channels", "out_channels", "dynamic_padding"]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _int_tuple_2_t = 3,
        stride: _int_tuple_2_t = 1,
        padding: str = "",
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        num_experts: int = 4,
    ):
        super().__init__()

        self.in_channels = in_channels  # I
        self.out_channels = out_channels  # O
        self.kernel_size = to_2tuple(kernel_size)  # K
        self.stride = to_2tuple(stride)  # s
        padding_val, is_padding_dynamic = get_padding_value(
            padding, kernel_size, stride=stride, dilation=dilation
        )
        self.dynamic_padding = is_padding_dynamic
        self.padding = to_2tuple(padding_val)  # P
        self.dilation = to_2tuple(dilation)  # D
        self.groups = groups  # G
        self.num_experts = num_experts  # E

        # TODO: Adapt shape for MLX data format BHWC
        # self.weight_shape = (
        #     self.out_channels,
        #     self.in_channels // self.groups,
        # ) + self.kernel_size  # (O, I // G, K, K)
        self.weight_shape = (
            self.out_channels,
            self.kernel_size[0],
            self.kernel_size[1],
            self.in_channels // self.groups,
        )  # (O, K, K, I // G)

        weight_num_param = 1  # P == O*K*K*(I//G)

        for wd in self.weight_shape:
            weight_num_param *= wd

        self.weight = mx.zeros(shape=(self.num_experts, weight_num_param))  # (E, P)

        if bias:
            self.bias_shape = (self.out_channels,)  # (O,)
            self.bias = mx.zeros(shape=(self.num_experts, self.out_channels))  # (E, O)
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        # FIXME: Original init has `a=sqrt(5)`, which is not available in MLX.
        # FIXME: Not sure that the init is correct.
        init_weight = get_condconv_initializer(
            nn.init.he_uniform(),
            self.num_experts,
            self.weight_shape,
        )
        init_weight(self.weight)

        if self.bias is not None:
            fan_in = np.prod(self.weight_shape[1:])
            bound = 1 / math.sqrt(fan_in)
            init_bias = get_condconv_initializer(
                nn.init.uniform(low=-bound, high=bound),
                self.num_experts,
                self.bias_shape,
            )
            init_bias(self.bias)

    # (B, H, W, I), (B, E)
    def __call__(self, x: mx.array, routing_weights: mx.array) -> mx.array:
        b, h, w, c = x.shape

        # (B, E) @ (E, P) -> (B, O*K*K*(I//G))
        weight = mx.matmul(routing_weights, self.weight)

        # (B, O*K*K*(I//G)) -> (B*O, K, K, (I//G))
        weight = weight.reshape(
            b * self.out_channels,
            self.kernel_size[0],
            self.kernel_size[1],
            self.in_channels // self.groups,
        )

        bias = None
        if self.bias is not None:
            # (B, E) @ (E, O) -> (B, O)
            bias = mx.matmul(routing_weights, self.bias)

        # (1, H, W, B*I)
        x = x.transpose(1, 2, 0, 3).reshape(1, h, w, b * c)

        if self.dynamic_padding:
            out = conv2d_same(
                x,
                weight,
                bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups * b,
            )
        else:
            # (1, H, W, B*I) conv (B*O, K, K, I//G) -> (1, H, W, B*O)
            out = mx.conv2d(
                x,
                weight,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups * b,
            )
            if bias is not None:
                out = out + bias.reshape(b * self.out_channels)

        # (H, W, B, O)
        out = out.reshape(out.shape[1], out.shape[2], b, self.out_channels)

        # (B, H, W, O)
        return out.transpose(2, 0, 1, 3)

    # (B, H, W, C), (B, E)
    def forward(self, x: mx.array, routing_weights: mx.array) -> mx.array:
        B, C, H, W = x.shape

        # (B, E) @ (E, P) -> (B, P)
        weight = mx.matmul(routing_weights, self.weight)

        # (B*O, I//G, K, K)
        new_weight_shape = (
            B * self.out_channels,
            self.in_channels // self.groups,
        ) + self.kernel_size

        # (B, Ox(I//G)*K*K) -> (B*O, I//G, K, K)
        weight = weight.reshape(new_weight_shape)

        bias = None
        if self.bias is not None:
            # (B, E) @ (E, O) -> (B, O)
            bias = mx.matmul(routing_weights, self.bias)

            # (B, O) -> (B*O,)
            bias = bias.reshape(B * self.out_channels)
        # move batch elements with channels so each batch element can be efficiently convolved with separate kernel
        # reshape instead of view to work with channels_last input

        # (1, B*C, H, W)
        x = x.reshape(1, B * C, H, W)

        if self.dynamic_padding:
            out = conv2d_same(
                x,
                weight,
                bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups * B,
            )
        else:
            # (1, B*C, H, W), (B*O, I//G, K, K) -> (1, B*O, H, W)
            out = mx.conv2d(
                x,
                weight,
                bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups * B,
            )

        # (1, B*O, H, W) -> (B, O, H, W)
        out = out.transpose(1, 0, 2, 3).reshape(
            B, self.out_channels, out.shape[-2], out.shape[-1]
        )

        return out
