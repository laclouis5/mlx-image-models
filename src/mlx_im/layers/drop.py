import mlx.core as mx
from mlx import nn

from .grid import ndgrid


def drop_block_2d(
    x: mx.array,
    drop_prob: float = 0.1,
    block_size: int = 7,
    gamma_scale: float = 1.0,
    with_noise: bool = False,
    inplace: bool = False,
    batchwise: bool = False,
) -> mx.array:
    _, H, W, C = x.shape
    total_size: int = W * H
    clipped_block_size = min(block_size, min(W, H))

    gamma: float = (
        gamma_scale
        * drop_prob
        * total_size
        / clipped_block_size**2
        / ((W - block_size + 1) * (H - block_size + 1))
    )

    w_i, h_i = ndgrid(mx.arange(W), mx.arange(H))
    valid_block = (
        (w_i >= clipped_block_size // 2) & (w_i < W - (clipped_block_size - 1) // 2)
    ) & ((h_i >= clipped_block_size // 2) & (h_i < H - (clipped_block_size - 1) // 2))

    valid_block = mx.reshape(valid_block, (1, H, W, 1))

    if batchwise:
        uniform_noise = mx.random.uniform(shape=(1, H, W, C), dtype=x.dtype)
    else:
        uniform_noise = mx.random.uniform(shape=x.shape, dtype=x.dtype)

    block_mask = ((2 - gamma - valid_block + uniform_noise) >= 1).astype(dtype=x.dtype)
    block_mask: mx.array = -nn.MaxPool2d(
        kernel_size=clipped_block_size,
        stride=1,
        padding=clipped_block_size // 2,
    )(-block_mask)

    if with_noise:
        normal_noise = (
            mx.random.normal(shape=(1, H, W, C), dtype=x.dtype)
            if batchwise
            else mx.random.normal(shape=x.shape, dtype=x.dtype)
        )
        x = x * block_mask + normal_noise * (1 - block_mask)
    else:
        normalize_scale = (
            block_mask.size / (block_mask.astype(dtype=mx.float32).sum() + 1e-7)
        ).astype(x.dtype)
        x = x * block_mask * normalize_scale

    return x


def drop_block_fast_2d(
    x: mx.array,
    drop_prob: float = 0.1,
    block_size: int = 7,
    gamma_scale: float = 1.0,
    with_noise: bool = False,
    inplace: bool = False,
) -> mx.array:
    _, H, W, _ = x.shape
    total_size: int = W * H
    clipped_block_size = min(block_size, min(W, H))

    gamma: float = (
        gamma_scale
        * drop_prob
        * total_size
        / clipped_block_size**2
        / ((W - block_size + 1) * (H - block_size + 1))
    )

    block_mask = mx.random.bernoulli(p=gamma, shape=x.shape).astype(dtype=x.dtype)
    block_mask: mx.array = nn.MaxPool2d(
        kernel_size=clipped_block_size,
        stride=1,
        padding=clipped_block_size // 2,
    )(block_mask)

    if with_noise:
        normal_noise = mx.random.normal(shape=x.shape, dtype=x.dtype)
        x = x * (1.0 - block_mask) + normal_noise * block_mask
    else:
        block_mask = 1 - block_mask
        normalize_scale = (
            block_mask.size / (block_mask.astype(dtype=mx.float32).sum() + 1e-6)
        ).astype(dtype=x.dtype)
        x = x * block_mask * normalize_scale
    return x


class DropBlock2d(nn.Module):
    def __init__(
        self,
        drop_prob: float = 0.1,
        block_size: int = 7,
        gamma_scale: float = 1.0,
        with_noise: bool = False,
        inplace: bool = False,
        batchwise: bool = False,
        fast: bool = True,
    ):
        super().__init__()
        self.drop_prob = drop_prob
        self.gamma_scale = gamma_scale
        self.block_size = block_size
        self.with_noise = with_noise
        self.inplace = inplace
        self.batchwise = batchwise
        self.fast = fast

    def __call__(self, x: mx.array) -> mx.array:
        if not self.training or not self.drop_prob:
            return x

        if self.fast:
            return drop_block_fast_2d(
                x,
                self.drop_prob,
                self.block_size,
                self.gamma_scale,
                self.with_noise,
                self.inplace,
            )
        else:
            return drop_block_2d(
                x,
                self.drop_prob,
                self.block_size,
                self.gamma_scale,
                self.with_noise,
                self.inplace,
                self.batchwise,
            )


def drop_path(
    x: mx.array,
    drop_prob: float = 0.0,
    training: bool = False,
    scale_by_keep: bool = True,
) -> mx.array:
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = mx.random.bernoulli(keep_prob, shape=shape).astype(x.dtype)

    if keep_prob > 0.0 and scale_by_keep:
        random_tensor = random_tensor / keep_prob

    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def __call__(self, x: mx.array) -> mx.array:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"
