import mlx.core as mx
import numpy as np
import timm.layers.grid as timm_m
import torch

import mlx_im.layers.grid as mlx_m


def test_meshgrid():
    x_mlx = [mx.array([0, 1, 2, 3, 4], dtype=mx.int32), mx.array([0, 1, 2])]
    x_torch = [torch.from_numpy(np.array(a)) for a in x_mlx]

    out_mlx = mlx_m.meshgrid(*x_mlx)
    out_timm = timm_m.meshgrid(*x_torch)

    mx.eval(out_mlx)

    for a1, a2 in zip(out_mlx, out_timm):
        a1 = np.array(a1)
        a2 = a2.detach().numpy()

        assert np.all(a1 == a2)


def test_ndgrid():
    x_mlx = [mx.array([0, 1, 2, 3, 4], dtype=mx.int32), mx.array([0, 1, 2])]
    x_torch = [torch.from_numpy(np.array(a)) for a in x_mlx]

    out_mlx = mlx_m.ndgrid(*x_mlx)
    out_timm = timm_m.ndgrid(*x_torch)

    mx.eval(out_mlx)

    for a1, a2 in zip(out_mlx, out_timm):
        a1 = np.array(a1)
        a2 = a2.detach().numpy()

        assert np.all(a1 == a2)
