from typing import Callable

import mlx.core as mx
from mlx import nn

from .conv_bn_act import ConvNormAct
from .create_act import create_act_layer
from .helpers import make_divisible


class ChannelAttn(nn.Module):
    def __init__(
        self,
        channels: int,
        rd_ratio: float = 1.0 / 16,
        rd_channels: int | None = None,
        rd_divisor: int = 1,
        act_layer: Callable[[], nn.Module] = nn.ReLU,
        gate_layer: str | type[nn.Module] | None = "sigmoid",
        mlp_bias: bool = False,
    ):
        super().__init__()
        if not rd_channels:
            rd_channels = make_divisible(
                channels * rd_ratio, rd_divisor, round_limit=0.0
            )
        self.fc1 = nn.Conv2d(channels, rd_channels, 1, bias=mlp_bias)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(rd_channels, channels, 1, bias=mlp_bias)
        self.gate = create_act_layer(gate_layer)

    def __call__(self, x: mx.array) -> mx.array:
        x_avg = self.fc2(self.act(self.fc1(x.mean(axis=(1, 2), keepdims=True))))
        x_max = self.fc2(self.act(self.fc1(x.max(axis=(1, 2), keepdims=True))))
        return x * self.gate(x_avg + x_max)


class LightChannelAttn(ChannelAttn):
    def __init__(
        self,
        channels: int,
        rd_ratio: float = 1.0 / 16,
        rd_channels: int | None = None,
        rd_divisor: int = 1,
        act_layer: Callable[[], nn.Module] = nn.ReLU,
        gate_layer: str | type[nn.Module] | None = "sigmoid",
        mlp_bias: bool = False,
    ):
        super().__init__(
            channels, rd_ratio, rd_channels, rd_divisor, act_layer, gate_layer, mlp_bias
        )

    def __call__(self, x: mx.array) -> mx.array:
        x_pool = x.mean((1, 2), keepdims=True) + x.max((1, 2), keepdims=True)
        x_attn = self.fc2(self.act(self.fc1(0.5 * x_pool)))
        return x * mx.sigmoid(x_attn)


class SpatialAttn(nn.Module):
    def __init__(
        self, kernel_size: int = 7, gate_layer: str | type[nn.Module] | None = "sigmoid"
    ):
        super().__init__()
        self.conv = ConvNormAct(2, 1, kernel_size, apply_act=False)
        self.gate = create_act_layer(gate_layer)

    def __call__(self, x: mx.array) -> mx.array:
        x_attn = mx.concat(
            [x.mean(axis=-1, keepdims=True), x.max(axis=-1, keepdims=True)], axis=-1
        )
        x_attn = self.conv(x_attn)
        return x * self.gate(x_attn)


class LightSpatialAttn(nn.Module):
    def __init__(
        self, kernel_size: int = 7, gate_layer: str | type[nn.Module] | None = "sigmoid"
    ):
        super().__init__()
        self.conv = ConvNormAct(1, 1, kernel_size, apply_act=False)
        self.gate = create_act_layer(gate_layer)

    def __call__(self, x: mx.array) -> mx.array:
        x_attn = x.mean(axis=-1, keepdims=True) + x.max(axis=-1, keepdims=True)
        x_attn = self.conv(0.5 * x_attn)
        return x * self.gate(x_attn)


class CbamModule(nn.Module):
    def __init__(
        self,
        channels: int,
        rd_ratio: float = 1.0 / 16,
        rd_channels: int | None = None,
        rd_divisor: int = 1,
        spatial_kernel_size: int = 7,
        act_layer: Callable[[], nn.Module] = nn.ReLU,
        gate_layer: str | type[nn.Module] | None = "sigmoid",
        mlp_bias: bool = False,
    ):
        super().__init__()
        self.channel = ChannelAttn(
            channels,
            rd_ratio=rd_ratio,
            rd_channels=rd_channels,
            rd_divisor=rd_divisor,
            act_layer=act_layer,
            gate_layer=gate_layer,
            mlp_bias=mlp_bias,
        )
        self.spatial = SpatialAttn(spatial_kernel_size, gate_layer=gate_layer)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.channel(x)
        x = self.spatial(x)
        return x


class LightCbamModule(nn.Module):
    def __init__(
        self,
        channels: int,
        rd_ratio: float = 1.0 / 16,
        rd_channels: int | None = None,
        rd_divisor: int = 1,
        spatial_kernel_size: int = 7,
        act_layer: Callable[[], nn.Module] = nn.ReLU,
        gate_layer: str | type[nn.Module] | None = "sigmoid",
        mlp_bias: bool = False,
    ):
        super().__init__()
        self.channel = LightChannelAttn(
            channels,
            rd_ratio=rd_ratio,
            rd_channels=rd_channels,
            rd_divisor=rd_divisor,
            act_layer=act_layer,
            gate_layer=gate_layer,
            mlp_bias=mlp_bias,
        )
        self.spatial = LightSpatialAttn(spatial_kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.channel(x)
        x = self.spatial(x)
        return x
