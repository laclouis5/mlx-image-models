from typing import Tuple

from mlx import nn


class GroupNorm(nn.GroupNorm):
    def __init__(
        self,
        num_channels: int,
        num_groups: int = 32,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        # NOTE num_channels is swapped to first arg for consistency in swapping norm layers with BN.
        super().__init__(
            num_groups=num_groups,
            dims=num_channels,
            eps=eps,
            affine=affine,
            pytorch_compatible=True,
        )


class GroupNorm1(GroupNorm):
    def __init__(self, num_channels: int, **kwargs):
        super().__init__(num_groups=1, num_channels=num_channels, **kwargs)


class LayerNorm(nn.LayerNorm):
    def __init__(self, num_channels: int, eps: float = 1e-6, affine: bool = True):
        super().__init__(dims=num_channels, eps=eps, affine=affine)


# NOTE: MLX LayerNorm is already on the last axis by default/
class LayerNorm2d(LayerNorm):
    pass


# NOTE: Let's say MLX is already well optimized.
class LayerNormExp2d(LayerNorm2d):
    pass


class RmsNorm(nn.RMSNorm):
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(
        self,
        channels: int,
        ep: float = 1e-6,
        affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(dims=channels, eps=ep)
        self.normalized_shape = (channels,)
        self.eps = ep
        self.elementwise_affine = affine


class RmsNorm2d(RmsNorm):
    pass
