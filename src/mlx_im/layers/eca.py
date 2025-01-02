import math

import mlx.core as mx
from mlx import nn

from .create_act import create_act_layer
from .helpers import make_divisible


class EcaModule(nn.Module):
    def __init__(
        self,
        channels: int | None = None,
        kernel_size: int = 3,
        gamma: float = 2,
        beta: float = 1,
        act_layer: str | type[nn.Module] | None = None,
        gate_layer: str | type[nn.Module] | None = "sigmoid",
        rd_ratio: float = 1 / 8,
        rd_channels: int | None = None,
        rd_divisor: int = 8,
        use_mlp: bool = False,
    ):
        super().__init__()

        if channels is not None:
            t = int(abs(math.log(channels, 2) + beta) / gamma)
            kernel_size = max(t if t % 2 else t + 1, 3)

        assert kernel_size % 2 == 1
        padding = (kernel_size - 1) // 2

        if use_mlp:
            # NOTE 'mlp' mode is a timm experiment, not in paper
            assert channels is not None
            if rd_channels is None:
                rd_channels = make_divisible(channels * rd_ratio, divisor=rd_divisor)
            act_layer = act_layer or nn.ReLU
            self.conv = nn.Conv1d(1, rd_channels, kernel_size=1, padding=0, bias=True)
            self.act = create_act_layer(act_layer)
            self.conv2 = nn.Conv1d(
                rd_channels, 1, kernel_size=kernel_size, padding=padding, bias=True
            )
        else:
            self.conv = nn.Conv1d(
                1, 1, kernel_size=kernel_size, padding=padding, bias=False
            )
            self.act = None
            self.conv2 = None

        self.gate = create_act_layer(gate_layer)

    def __call__(self, x: mx.array) -> mx.array:
        y = x.mean(axis=(1, 2))[:, :, None]
        y = self.conv(y)

        if self.conv2 is not None:
            y = self.act(y)
            y = self.conv2(y)

        y = self.gate(y)
        y = y.reshape(x.shape[0], 1, 1, -1)
        return x * y


EfficientChannelAttn = EcaModule


class CecaModule(nn.Module):
    def __init__(
        self,
        channels: int | None = None,
        kernel_size: int = 3,
        gamma: float = 2,
        beta: float = 1,
        act_layer: str | type[nn.Module] | None = None,
        gate_layer: str | type[nn.Module] | None = "sigmoid",
    ):
        super().__init__()

        if channels is not None:
            t = int(abs(math.log(channels, 2) + beta) / gamma)
            kernel_size = max(t if t % 2 else t + 1, 3)

        has_act = act_layer is not None
        assert kernel_size % 2 == 1

        self.padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=0, bias=has_act)
        self.gate = create_act_layer(gate_layer)

    @staticmethod
    def _pad_circular_1d(x: mx.array, pad_width: tuple[int, int]) -> mx.array:
        # NOTE Unsure if necessary to stop gradients in the padded zone.
        left = mx.stop_gradient(x[:, -pad_width[1] :, :])
        right = mx.stop_gradient(x[:, : pad_width[0], :])
        return mx.concat([left, x, right], axis=1)

    def __call__(self, x: mx.array) -> mx.array:
        y = x.mean((1, 2)).reshape(x.shape[0], -1, 1)
        y = self._pad_circular_1d(y, (self.padding, self.padding))

        y = self.conv(y)
        y = self.gate(y).reshape(x.shape[0], 1, 1, -1)

        return x * y


CircularEfficientChannelAttn = CecaModule
