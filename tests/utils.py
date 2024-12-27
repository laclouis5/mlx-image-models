import mlx.core as mx
import numpy as np
import torch


def torch_to_mlx_2d(x: torch.Tensor) -> mx.array:
    x = x.permute(0, 2, 3, 1)
    return mx.array(x.detach().numpy())


def mlx_to_numpy_2d(x: mx.array) -> np.ndarray:
    x = x.transpose(0, 3, 1, 2)
    return np.array(x)


def mlx_to_torch_2d(x: mx.array) -> torch.Tensor:
    return torch.from_numpy(mlx_to_numpy_2d(x))


def sample_mlx_array_2d(shape: tuple[int, int, int, int]) -> mx.array:
    return mx.random.normal(
        shape=shape, dtype=mx.float32, key=mx.array([42, 42], dtype=mx.uint32)
    )
