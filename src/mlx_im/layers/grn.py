import mlx.core as mx
from mlx import nn


class GlobalResponseNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, channels_last: bool = True):
        super().__init__()
        self.eps = eps

        if channels_last:
            self.spatial_dim = (1, 2)
            self.channel_dim = -1
            self.wb_shape = (1, 1, 1, -1)
        else:
            self.spatial_dim = (2, 3)
            self.channel_dim = 1
            self.wb_shape = (1, -1, 1, 1)

        self.weight = mx.zeros(dim)
        self.bias = mx.zeros(dim)

    def __call__(self, x: mx.array) -> mx.array:
        x_g = mx.linalg.norm(x, axis=self.spatial_dim, keepdims=True)
        x_n = x_g / (x_g.mean(axis=self.channel_dim, keepdims=True) + self.eps)
        return (
            x
            + self.bias.reshape(self.wb_shape)
            + self.weight.reshape(self.wb_shape) * x * x_n
        )
