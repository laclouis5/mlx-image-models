from typing import Callable

import mlx.core as mx
from mlx import nn

from .create_act import create_act_layer
from .helpers import make_divisible


class SEModule(nn.Module):
    def __init__(
        self,
        channels: int,
        rd_ratio: float = 1.0 / 16,
        rd_channels: int | None = None,
        rd_divisor: int = 8,
        add_maxpool: bool = False,
        bias: bool = True,
        act_layer: str | type[nn.Module] | None = nn.ReLU,
        norm_layer: str | Callable[[int], nn.Module] | None = None,
        gate_layer: str | type[nn.Module] | None = "sigmoid",
    ):
        super().__init__()

        self.add_maxpool = add_maxpool

        if not rd_channels:
            rd_channels = make_divisible(
                channels * rd_ratio, rd_divisor, round_limit=0.0
            )

        self.fc1 = nn.Conv2d(channels, rd_channels, kernel_size=1, bias=bias)
        self.bn = norm_layer(rd_channels) if norm_layer else nn.Identity()
        self.act = create_act_layer(act_layer, inplace=True)
        self.fc2 = nn.Conv2d(rd_channels, channels, kernel_size=1, bias=bias)
        self.gate = create_act_layer(gate_layer)

    def __call__(self, x: mx.array) -> mx.array:
        x_se = x.mean(axis=(1, 2), keepdims=True)

        if self.add_maxpool:
            x_se = 0.5 * (x_se + x.max(axis=(1, 2), keepdims=True))

        x_se = self.fc1(x_se)
        x_se = self.act(self.bn(x_se))
        x_se = self.fc2(x_se)

        return x * self.gate(x_se)


SqueezeExcite = SEModule


class EffectiveSEModule(nn.Module):
    def __init__(
        self,
        channels: int,
        add_maxpool: bool = False,
        gate_layer: str | type[nn.Module] | None = "hard_sigmoid",
        **_,
    ):
        super().__init__()

        self.add_maxpool = add_maxpool
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.gate = create_act_layer(gate_layer)

    def __call__(self, x: mx.array) -> mx.array:
        x_se = x.mean(axis=(1, 2), keepdims=True)

        if self.add_maxpool:
            x_se = 0.5 * (x_se + x.max(axis=(1, 2), keepdims=True))

        x_se = self.fc(x_se)

        return x * self.gate(x_se)


EffectiveSqueezeExcite = EffectiveSEModule


class SqueezeExciteCl(nn.Module):
    def __init__(
        self,
        channels: int,
        rd_ratio: float = 1.0 / 16,
        rd_channels: int | None = None,
        rd_divisor: int = 8,
        bias: bool = True,
        act_layer: str | type[nn.Module] | None = nn.ReLU,
        gate_layer: str | type[nn.Module] | None = "sigmoid",
    ):
        super().__init__()

        if not rd_channels:
            rd_channels = make_divisible(
                channels * rd_ratio, rd_divisor, round_limit=0.0
            )

        self.fc1 = nn.Linear(channels, rd_channels, bias=bias)
        self.act = create_act_layer(act_layer, inplace=True)
        self.fc2 = nn.Linear(rd_channels, channels, bias=bias)
        self.gate = create_act_layer(gate_layer)

    def __call__(self, x: mx.array) -> mx.array:
        x_se = x.mean(axis=(1, 2), keepdims=True)
        x_se = self.fc1(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)
