import math
from typing import Callable

import mlx.core as mx
from mlx import nn

from .create_act import create_act_layer, get_act_layer
from .create_conv2d import create_conv2d
from .helpers import make_divisible
from .mlp import ConvMlp
from .mlx_layers import AvgPool2d


class GatherExcite(nn.Module):
    def __init__(
        self,
        channels: int,
        feat_size: int | None = None,
        extra_params: bool = False,
        extent: int = 0,
        use_mlp: bool = True,
        rd_ratio: float = 1.0 / 16,
        rd_channels: int | None = None,
        rd_divisor: int = 1,
        add_maxpool: bool = False,
        act_layer: Callable[[], nn.Module] = nn.ReLU,
        norm_layer: Callable[[int], nn.Module] | None = nn.BatchNorm,
        gate_layer: str | type[nn.Module] | None = "sigmoid",
    ):
        super().__init__()

        self.add_maxpool = add_maxpool
        act_layer = get_act_layer(act_layer)
        self.extent = extent

        if extra_params:
            gather = []
            if extent == 0:
                assert (
                    feat_size is not None
                ), "spatial feature size must be specified for global extent w/ params"
                gather.append(
                    create_conv2d(
                        channels,
                        channels,
                        kernel_size=feat_size,
                        stride=1,
                        depthwise=True,
                    ),
                )
                if norm_layer is not None:
                    gather.append(norm_layer(channels))
            else:
                assert extent % 2 == 0
                num_conv = int(math.log2(extent))
                for i in range(num_conv):
                    gather.append(
                        create_conv2d(
                            channels, channels, kernel_size=3, stride=2, depthwise=True
                        ),
                    )
                    if norm_layer:
                        gather.append(norm_layer(channels))
                    if i != num_conv - 1:
                        gather.append(act_layer(inplace=True))

            self.gather = nn.Sequential(*gather)
        else:
            self.gather = None
            if self.extent == 0:
                self.gk = 0
                self.gs = 0
            else:
                assert extent % 2 == 0
                self.gk = self.extent * 2 - 1
                self.gs = self.extent

        if not rd_channels:
            rd_channels = make_divisible(
                channels * rd_ratio, rd_divisor, round_limit=0.0
            )
        self.mlp = (
            ConvMlp(channels, rd_channels, act_layer=act_layer)
            if use_mlp
            else nn.Identity()
        )
        self.gate = create_act_layer(gate_layer)

    def __call__(self, x: mx.array) -> mx.array:
        if self.gather is not None:
            x_ge = self.gather(x)
        else:
            if self.extent == 0:
                x_ge = x.mean(axis=(1, 2), keepdims=True)
                if self.add_maxpool:
                    x_ge = 0.5 * x_ge + 0.5 * x.max((1, 2), keepdims=True)
            else:
                x_ge = AvgPool2d(
                    kernel_size=self.gk,
                    stride=self.gs,
                    padding=self.gk // 2,
                    count_include_pad=False,
                )(x)
                if self.add_maxpool:
                    x_ge = 0.5 * x_ge + 0.5 * nn.MaxPool2d(
                        kernel_size=self.gk, stride=self.gs, padding=self.gk // 2
                    )(x)

        x_ge = self.mlp(x_ge)

        if x_ge.shape[1] != 1 or x_ge.shape[2] != 1:
            scale_factor = x.shape[1] // x_ge.shape[1], x.shape[2] // x_ge.shape[2]
            x_ge = nn.Upsample(scale_factor=scale_factor)(x_ge)

        return x * self.gate(x_ge)
