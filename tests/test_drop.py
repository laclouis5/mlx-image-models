import mlx.core as mx
import numpy as np
import pytest
import timm.layers.drop as timm_m

import mlx_im.layers.drop as mlx_m

from . import utils as U


@pytest.mark.parametrize("drop_prob", [0.0, 0.1])
@pytest.mark.parametrize("batchwise", [True, False])
@pytest.mark.parametrize("with_noise", [True, False])
@pytest.mark.parametrize("fast", [True, False])
@pytest.mark.parametrize("training", [True, False])
def test_drop_block(drop_prob, batchwise, with_noise, fast, training):
    x_mlx = U.sample_mlx_array_2d(shape=(2, 32, 48, 32))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_m.DropBlock2d(
        drop_prob=drop_prob, batchwise=batchwise, with_noise=with_noise, fast=fast
    )
    mod_timm = timm_m.DropBlock2d(
        drop_prob=drop_prob, batchwise=batchwise, with_noise=with_noise, fast=fast
    )

    mod_mlx = mod_mlx.train(training)
    mod_timm = mod_timm.train(training)

    out_mlx = mod_mlx(x_mlx)
    out_timm = mod_timm(x_torch)

    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    if drop_prob == 0.0 or not training:
        assert np.allclose(
            out_mlx, out_timm, atol=1.0e-5
        ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"
    else:
        assert out_mlx.shape == tuple(out_timm.shape)


@pytest.mark.parametrize("drop_prob", [0.0, 0.1])
@pytest.mark.parametrize("training", [True, False])
def test_drop_path(drop_prob, training):
    x_mlx = U.sample_mlx_array_2d(shape=(2, 32, 48, 32))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_m.DropPath(drop_prob=drop_prob)
    mod_timm = timm_m.DropPath(drop_prob=drop_prob)

    mod_mlx = mod_mlx.train(training)
    mod_timm = mod_timm.train(training)

    out_mlx = mod_mlx(x_mlx)
    out_timm = mod_timm(x_torch)

    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    if drop_prob == 0.0 or not training:
        assert np.allclose(
            out_mlx, out_timm, atol=1.0e-5
        ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"
    else:
        assert out_mlx.shape == tuple(out_timm.shape)
