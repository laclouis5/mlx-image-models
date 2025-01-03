from typing import Callable

import mlx.core as mx
from mlx import nn

from .create_act import create_act_layer, get_act_layer
from .helpers import make_divisible
from .mlp import ConvMlp
from .norm import LayerNorm2d


class GlobalContext(nn.Module):
    def __init__(
        self,
        channels: int,
        use_attn: bool = True,
        fuse_add: bool = False,
        fuse_scale: bool = True,
        init_last_zero: bool = False,
        rd_ratio: float = 1.0 / 8,
        rd_channels: int | None = None,
        rd_divisor: int = 1,
        act_layer: str | Callable[[], nn.Module] | None = nn.ReLU,
        gate_layer: str | type[nn.Module] | None = "sigmoid",
    ):
        super().__init__()
        act_layer = get_act_layer(act_layer)

        self.conv_attn = (
            nn.Conv2d(channels, 1, kernel_size=1, bias=True) if use_attn else None
        )

        if rd_channels is None:
            rd_channels = make_divisible(
                channels * rd_ratio, rd_divisor, round_limit=0.0
            )
        if fuse_add:
            self.mlp_add = ConvMlp(
                channels, rd_channels, act_layer=act_layer, norm_layer=LayerNorm2d
            )
        else:
            self.mlp_add = None
        if fuse_scale:
            self.mlp_scale = ConvMlp(
                channels, rd_channels, act_layer=act_layer, norm_layer=LayerNorm2d
            )
        else:
            self.mlp_scale = None

        self.gate = create_act_layer(gate_layer)

    def __call__(self, x: mx.array) -> mx.array:
        B, H, W, C = x.shape

        if self.conv_attn is not None:
            attn = self.conv_attn(x).reshape(B, 1, 1, H * W)  # (B, 1, 1, H * W)
            attn = mx.softmax(attn, axis=-1)  # (B, 1, 1, H * W)

            # (B, H*W, C) -> (B, 1, 1, H * W) @ (B, 1, H*W, C) -> (B, 1, 1, C)
            context = attn @ x.reshape(B, -1, H * W, C)
        else:
            # (B, 1, 1, C)
            context = x.mean(dim=(1, 2), keepdim=True)

        if self.mlp_scale is not None:
            mlp_x = self.mlp_scale(context)
            x = x * self.gate(mlp_x)

        if self.mlp_add is not None:
            mlp_x = self.mlp_add(context)
            x = x + mlp_x

        return x
